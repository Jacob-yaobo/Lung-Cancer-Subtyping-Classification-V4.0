# scripts/test.py

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# 导入你的自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import LungCancerDataset
from src.transforms import get_transforms
from src.model import HierarchicalFusionResNet
from src.utils import get_metrics, plot_confusion_matrix, plot_roc_curves

def test(model, data_loader, device, num_classes, class_names, output_dir):
    """
    在指定数据集上测试模型，并生成详细的性能报告。
    """
    model.eval()
    
    all_subject_ids = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing", unit="batch"):
            inputs_dict, labels, subject_ids = batch
            
            inputs_dict = {key: tensor.to(device) for key, tensor in inputs_dict.items()}
            outputs = model(**inputs_dict)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_subject_ids.extend(subject_ids)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs_cat = torch.cat(all_probs)
    all_labels_cat = torch.cat(all_labels)
    all_preds_cat = torch.argmax(all_probs_cat, dim=1)

    # --- 1. Slice级别分析 ---
    print("\n--- Calculating Slice-Level Metrics ---")
    slice_metrics = get_metrics(all_labels_cat.numpy(), all_preds_cat.numpy(), all_probs_cat.numpy(), num_classes)
    plot_confusion_matrix(slice_metrics['confusion_matrix'], class_names, 
                          'Slice-Level Confusion Matrix', os.path.join(output_dir, 'cm_slice.png'))
    plot_roc_curves(all_labels_cat.numpy(), all_probs_cat.numpy(), num_classes, class_names,
                    'Slice-Level ROC Curves (One-vs-Rest)', os.path.join(output_dir, 'roc_slice.png'))
    print("Slice-level analysis complete.")

    # --- 2. Patient级别分析 ---
    print("\n--- Calculating Patient-Level Metrics ---")
    df_data = {'subject_id': all_subject_ids, 'true_label': all_labels_cat.numpy()}
    for i in range(num_classes):
        df_data[f'prob_{i}'] = all_probs_cat[:, i].numpy()
    df = pd.DataFrame(df_data)
    
    patient_df = df.groupby('subject_id').mean() # 对一个patient的所有slice概率取平均（采用了probability average pooling的方式）
    
    patient_true_labels = patient_df['true_label'].astype(int).values
    patient_prob_cols = [f'prob_{i}' for i in range(num_classes)]
    patient_probs = patient_df[patient_prob_cols].values
    patient_preds = np.argmax(patient_probs, axis=1)
    
    patient_metrics = get_metrics(patient_true_labels, patient_preds, patient_probs, num_classes)
    plot_confusion_matrix(patient_metrics['confusion_matrix'], class_names, 
                          'Patient-Level Confusion Matrix', os.path.join(output_dir, 'cm_patient.png'))
    plot_roc_curves(patient_true_labels, patient_probs, num_classes, class_names,
                    'Patient-Level ROC Curves (One-vs-Rest)', os.path.join(output_dir, 'roc_patient.png'))
    print("Patient-level analysis complete.")

    # --- 3. 保存所有指标到文件 ---
    full_report = {
        'slice_level': slice_metrics,
        'patient_level': patient_metrics
    }
    report_path = os.path.join(output_dir, 'full_metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    print(f"\nFull report saved to {report_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 加载数据 ---
    # 在测试时，我们通常在验证集上进行详细评估
    val_transform = get_transforms(mode='val')
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
    data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    task_info = val_dataset.get_task_info()
    num_classes = len(task_info['labels'])
    # 获取类别名称用于绘图，需要按标签索引排序
    class_names = [name for name, index in sorted(task_info['labels'].items(), key=lambda item: item[1])]
    
    # --- 加载模型 ---
    model = HierarchicalFusionResNet(num_subtypes=num_classes, pretrained=False).to(device)
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # --- 执行测试 ---
    test(model, data_loader, device, num_classes, class_names, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lung Cancer Subtype Classification Testing")
    
    # 核心参数
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth) file')
    parser.add_argument('--fold', type=int, required=True, help='Which validation fold data to use (must match the trained model)')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task')
    parser.add_argument('--output_dir', type=str, default='./results/test_report', help='Directory to save test reports and plots')

    # 数据路径参数 (与train.py保持一致)
    parser.add_argument('--data_dir', type=str, default='./data/2_final_h5/')
    parser.add_argument('--participants_path', type=str, default='./metadata/participants.tsv')
    parser.add_argument('--tasks_path', type=str, default='./metadata/tasks.json')
    parser.add_argument('--splits_path', type=str, default='./metadata/splits.json')
    parser.add_argument('--lesion_info_path', type=str, default='./metadata/lesion_slice_info.json')

    # DataLoader参数
    parser.add_argument('--batch_size', type=int, default=32) # 测试时可以使用更大的batch_size
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)