# scripts/train.py

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd

# 导入你的自定义模块
# 假设train.py在scripts/目录下，需要将项目根目录添加到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import LungCancerDataset
from src.transforms import get_transforms
from src.model import HierarchicalFusionResNet
from src.utils import set_seed, AverageMeter, get_metrics

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch。"""
    model.train()
    loss_meter = AverageMeter()
    
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    for batch in progress_bar:
        inputs_dict, labels, _ = batch  # 忽略 subject_id
        
        # 将数据移动到GPU
        inputs_dict = {key: tensor.to(device) for key, tensor in inputs_dict.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs_dict)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), labels.size(0))
        progress_bar.set_postfix(loss=loss_meter.avg)
        
    return loss_meter.avg

def validate(model, val_loader, device, num_classes):
    """在验证集上评估模型，并同时计算slice和patient级别指标。"""
    model.eval()
    
    all_subject_ids = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", unit="batch"):
            inputs_dict, labels, subject_ids = batch
            
            inputs_dict = {key: tensor.to(device) for key, tensor in inputs_dict.items()}
            outputs = model(**inputs_dict)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_subject_ids.extend(subject_ids)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    # --- 1. 计算 Slice 级别指标 ---
    all_probs_cat = torch.cat(all_probs)
    all_labels_cat = torch.cat(all_labels)
    all_preds_cat = torch.argmax(all_probs_cat, dim=1)
    
    slice_metrics = get_metrics(all_labels_cat.numpy(), all_preds_cat.numpy(), all_probs_cat.numpy(), num_classes)
    
    # --- 2. 计算 Patient 级别指标 (Soft Voting) ---
    df_data = {'subject_id': all_subject_ids, 'true_label': all_labels_cat.numpy()}
    for i in range(num_classes):
        df_data[f'prob_{i}'] = all_probs_cat[:, i].numpy()
        
    df = pd.DataFrame(df_data)
    
    # 按病人ID分组，并对概率求平均
    patient_df = df.groupby('subject_id').mean()
    
    patient_true_labels = patient_df['true_label'].astype(int)
    patient_prob_cols = [f'prob_{i}' for i in range(num_classes)]
    patient_probs = patient_df[patient_prob_cols].values
    patient_preds = np.argmax(patient_probs, axis=1)
    
    patient_metrics = get_metrics(patient_true_labels, patient_preds, patient_probs, num_classes)
    
    return {'slice': slice_metrics, 'patient': patient_metrics}


def main(args):
    # --- 1. 环境与配置 ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存本次运行的配置
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # --- 2. 数据加载 ---
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')

    train_dataset = LungCancerDataset(
        data_dir=args.data_dir,
        participants_path=args.participants_path,
        tasks_path=args.tasks_path,
        splits_path=args.splits_path,
        lesion_info_path=args.lesion_info_path,
        task_name=args.task_name,
        fold=args.fold,
        mode='train',
        transform=train_transform
    )
    val_dataset = LungCancerDataset(
        data_dir=args.data_dir,
        participants_path=args.participants_path,
        tasks_path=args.tasks_path,
        splits_path=args.splits_path,
        lesion_info_path=args.lesion_info_path,
        task_name=args.task_name,
        fold=args.fold,
        mode='val',
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # --- 3. 模型、损失函数、优化器 ---
    num_classes = len(train_dataset.get_task_info()['labels'])
    model = HierarchicalFusionResNet(num_subtypes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # --- 4. 日志与训练循环 ---
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    best_metric = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device, num_classes)
        
        scheduler.step()
        
        # --- 日志记录 ---
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
        print(f"  Slice-Level -> Acc: {val_metrics['slice']['accuracy']:.4f} | AUC: {val_metrics['slice'].get('auc_ovr_weighted', 0):.4f}")
        print(f"  Patient-Level -> Acc: {val_metrics['patient']['accuracy']:.4f} | AUC: {val_metrics['patient'].get('auc_ovr_weighted', 0):.4f}")

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        for level in ['slice', 'patient']:
            for metric, value in val_metrics[level].items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Metrics_{level}/{metric}', value, epoch)
        
        # --- 模型保存 ---
        # 以 patient 级别的 AUC 作为核心指标
        current_metric = val_metrics['patient'].get('auc_ovr_weighted', val_metrics['patient']['accuracy'])
        if current_metric > best_metric:
            best_metric = current_metric
            print(f"🎉 New best model found at epoch {epoch} with Patient AUC: {best_metric:.4f}")
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_fold_{args.fold}.pth'))

    writer.close()
    print("\n--- Training finished! ---")
    print(f"Best Patient-Level AUC for fold {args.fold}: {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lung Cancer Subtype Classification Training")
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='./data/2_final_h5/', help='Path to HDF5 data directory')
    parser.add_argument('--participants_path', type=str, default='./metadata/participants.tsv')
    parser.add_argument('--tasks_path', type=str, default='./metadata/tasks.json')
    parser.add_argument('--splits_path', type=str, default='./metadata/splits.json')
    parser.add_argument('--lesion_info_path', type=str, default='./metadata/lesion_slice_info.json')
    parser.add_argument('--output_dir', type=str, default='./results/run1', help='Directory to save outputs')
    
    # 训练参数
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task defined in tasks.json')
    parser.add_argument('--fold', type=int, required=True, help='Which fold to use for validation (0-4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)