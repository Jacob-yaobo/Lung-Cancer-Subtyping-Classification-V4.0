"""
LungCancerDataset模块
用于肺癌亚型分类的PyTorch数据集类，支持多任务配置、分层交叉验证和2.5D切片提取。
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List, Dict, Any


class LungCancerDataset(Dataset):
    """
    用于肺癌亚型分类的PyTorch数据集类。
    
    功能特性:
    - 支持多任务配置 (e.g., 三分类, 二分类)
    - 支持分层k折交叉验证
    - 从3D HDF5数据中提取2.5D切片 (每个模态提取前、中、后三层切片)
    - 通过可组合的transform实现归一化、ToTensor和数据增强
    
    数据格式:
    - HDF5文件包含 'CT', 'PET', 'Lesion_mask' 三个数据集
    - 维度顺序: (H, W, D)
    - 输出格式: 字典形式，包含三个模态
        - 'ct': (3, H, W) - [prev, center, next]
        - 'pet': (3, H, W) - [prev, center, next]
        - 'mask': (3, H, W) - [prev, center, next]
    """
    
    def __init__(
        self,
        data_dir: str,
        participants_path: str,
        tasks_path: str,
        splits_path: str,
        lesion_info_path: str,
        task_name: str,
        fold: int,
        mode: str = 'train',
        augmentations: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        """
        初始化LungCancerDataset。

        Args:
            data_dir (str): 存放HDF5文件的目录路径 (e.g., 'data/2_final_h5/')
            participants_path (str): 参与者信息TSV文件路径 (e.g., 'metadata/participants.tsv')
            tasks_path (str): 任务定义JSON文件路径 (e.g., 'metadata/tasks.json')
            splits_path (str): 交叉验证划分JSON文件路径 (e.g., 'metadata/splits.json')
            lesion_info_path (str): 病灶切片索引JSON文件路径 (e.g., 'metadata/lesion_slice_info.json')
            task_name (str): 任务名称，必须在tasks.json中定义 (e.g., '3-class_classification')
            fold (int): 交叉验证折数 (0-4)
            mode (str): 数据集模式，'train' 或 'val'
            augmentations (callable, optional): 数据增强函数（向后兼容），当transform未提供时会用于默认的LungCancerTransform
            transform (callable, optional): 样本级变换函数，接收{'modalities': dict, 'label': int}并返回同结构（常用来做归一化、ToTensor、数据增强等）
        
        Raises:
            FileNotFoundError: 当元数据文件不存在时
            KeyError: 当task_name不在tasks.json中时
            ValueError: 当fold超出范围或mode不合法时
        """
        super().__init__()
        
        # 保存基本参数
        self.data_dir = data_dir
        self.mode = mode
        self.task_name = task_name
        self.fold = fold
        if transform is not None:
            self.transform = transform
        else:
            self.transform = LungCancerTransform(augmentations=augmentations)
        
        # 参数验证
        if mode not in ['train', 'val']:
            raise ValueError(f"mode必须是'train'或'val'，当前值: {mode}")
        
        # --- 1. 加载所有元数据 ---
        print(f"[Dataset初始化] 加载元数据文件...")
        
        # 1.1 加载participants.tsv - 包含受试者ID和病理标签
        if not os.path.exists(participants_path):
            raise FileNotFoundError(f"未找到参与者信息文件: {participants_path}")
        self.participants_df = pd.read_csv(participants_path, sep='\t')
        print(f"  ✓ 加载参与者信息: {len(self.participants_df)} 个受试者")
        
        # 1.2 加载tasks.json - 定义分类任务和标签映射
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(f"未找到任务配置文件: {tasks_path}")
        with open(tasks_path, 'r', encoding='utf-8') as f:
            self.tasks = json.load(f)
        print(f"  ✓ 加载任务配置: {list(self.tasks.keys())}")
        
        # 1.3 加载splits.json - 定义交叉验证划分
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"未找到划分文件: {splits_path}")
        with open(splits_path, 'r', encoding='utf-8') as f:
            self.splits = json.load(f)
        print(f"  ✓ 加载交叉验证划分: {len(self.splits)} 折")
        
        # 1.4 加载lesion_slice_info.json - 包含每个受试者的病灶切片索引
        if not os.path.exists(lesion_info_path):
            raise FileNotFoundError(f"未找到病灶切片信息文件: {lesion_info_path}")
        with open(lesion_info_path, 'r', encoding='utf-8') as f:
            self.lesion_slice_info = json.load(f)
        print(f"  ✓ 加载病灶切片信息: {len(self.lesion_slice_info)} 个受试者")
        
        # --- 2. 确定任务配置 ---
        # 在这一步，根据task的要求，选择完成二分类还是三分类，并确定标签映射
        print(f"\n[任务配置] 任务: {task_name}, 折数: {fold}, 模式: {mode}")
        
        # 2.1 获取当前任务的配置
        if task_name not in self.tasks:
            raise KeyError(f"任务 '{task_name}' 未在tasks.json中定义。可用任务: {list(self.tasks.keys())}")
        self.task_config = self.tasks[task_name]
        self.label_mapping = self.task_config['labels']
        print(f"  ✓ 标签映射: {self.label_mapping}")
        
        # 2.2 获取当前任务涉及的所有病理类型
        task_pathologies = set(self.label_mapping.keys())
        print(f"  ✓ 任务病理类型: {task_pathologies}")
        
        # --- 3. 确定训练/验证集划分 ---
        # 在splits.json中，以及按照center*label对受试者完成了分层k折划分，根据fold的设置，选择训练集和验证集。
        print(f"\n[数据划分] Fold {fold}, Mode: {mode}")
        
        # 3.1 验证fold编号
        if fold < 0 or fold >= len(self.splits):
            raise ValueError(f"fold必须在0到{len(self.splits)-1}之间，当前值: {fold}")
        
        # 3.2 根据mode参数，选择fold对应的列表为验证集，其余为训练集
        if mode == 'val':
            # 验证集: 直接使用当前fold的val列表
            current_fold_data = self.splits[fold]
            fold_subject_ids = current_fold_data['val']
            print(f"  ✓ 从Fold {fold}的val集获取: {len(fold_subject_ids)} 个受试者")
        else:  # mode == 'train'
            # 训练集: 合并除当前fold外的其他所有fold的val列表
            fold_subject_ids = []
            for i, fold_data in enumerate(self.splits):
                if i != fold:  # 排除当前fold
                    fold_subject_ids.extend(fold_data['val'])
            print(f"  ✓ 合并其他{len(self.splits)-1}个fold的val集: {len(fold_subject_ids)} 个受试者")

        # 3.3 基于fold的受试者筛选: 将splits列表中的受试者与participants.tsv进行交叉验证，确保ID存在且有标签信息
        # 构建participants中的subject_id到病理标签的映射
        participants_dict = {}
        for _, row in self.participants_df.iterrows():
            participants_dict[row['subject_id']] = row['Pathology']
        
        # 筛选出在participants.tsv中存在的subject_id
        filtered_subjects_with_labels = []
        missing_subjects = []
        for subject_id in fold_subject_ids:
            if subject_id in participants_dict:
                pathology = participants_dict[subject_id]
                filtered_subjects_with_labels.append((subject_id, pathology))
            else:
                missing_subjects.append(subject_id)
        
        if missing_subjects:
            print(f"  ⚠ 警告: {len(missing_subjects)} 个受试者在participants.tsv中未找到: {missing_subjects[:5]}...")
        print(f"  ✓ 第一次筛选后: {len(filtered_subjects_with_labels)} 个受试者有标签信息")

        # 3.4 基于task的筛选: 根据分类任务的类型要求，只保留需要的标签类型
        final_subjects_with_labels = []
        excluded_count = 0
        for subject_id, pathology in filtered_subjects_with_labels:
            if pathology in task_pathologies:
                final_subjects_with_labels.append((subject_id, pathology))
            else:
                excluded_count += 1
        
        if excluded_count > 0:
            print(f"  ✓ 第二次筛选: 排除了 {excluded_count} 个不在任务范围内的病理类型")
        print(f"  ✓ 最终{mode}集受试者数: {len(final_subjects_with_labels)}")
        
        # 保存最终的subject_id列表和标签信息
        self.subject_ids = [subj_id for subj_id, _ in final_subjects_with_labels]
        self.subject_pathologies = {subj_id: pathology for subj_id, pathology in final_subjects_with_labels}
        
        # 打印类别分布
        pathology_counts = {}
        for _, pathology in final_subjects_with_labels:
            pathology_counts[pathology] = pathology_counts.get(pathology, 0) + 1
        print(f"  ✓ 受试者病理分布: {pathology_counts}")
        
        # --- 4. 构建核心样本列表 (self.samples) ---
        print(f"\n[构建样本列表]")
        self.samples: List[Tuple[str, int, int]] = []
        
        # 统计信息
        subjects_with_lesion = 0
        subjects_without_lesion = 0
        total_slices = 0
        
        # 遍历所有最终的subject_id
        for subject_id in self.subject_ids:
            # 4.1 获取该subject的病理标签（已经在前面筛选过，保证存在）
            pathology = self.subject_pathologies[subject_id]
            
            # 4.2 将病理标签映射为数字标签（已经在前面筛选过，保证在label_mapping中）
            numeric_label = self.label_mapping[pathology]
            
            # 4.3 从lesion_slice_info中获取该subject的病灶切片索引列表
            if subject_id not in self.lesion_slice_info:
                # 某些受试者可能没有病灶切片信息
                subjects_without_lesion += 1
                continue
            
            lesion_slices = self.lesion_slice_info[subject_id]
            
            if not lesion_slices or len(lesion_slices) == 0:
                subjects_without_lesion += 1
                continue
            
            subjects_with_lesion += 1
            
            # 4.4 遍历该受试者的所有病灶切片索引，为每个切片创建一个样本
            for slice_idx in lesion_slices:
                # 将 (subject_id, slice_idx, label) 添加到样本列表
                self.samples.append((subject_id, slice_idx, numeric_label))
                total_slices += 1
        
        # 打印统计信息
        print(f"  ✓ 有病灶切片的受试者: {subjects_with_lesion}")
        print(f"  ✓ 无病灶切片的受试者: {subjects_without_lesion}")
        print(f"  ✓ 总样本数(切片数): {total_slices}")
        
        # 统计各类别样本数
        label_counts = {}
        for _, _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  ✓ 类别分布: {label_counts}")
        
        if len(self.samples) == 0:
            print(f"  ⚠ 警告: 数据集为空！请检查数据文件和配置。")
    
    def __len__(self) -> int:
        """
        返回数据集中样本的总数。
        
        Returns:
            int: 样本总数（即2.5D切片的数量）
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        根据索引获取一个2.5D样本。
        
        处理流程:
        1. 获取样本信息(subject_id, slice_idx, label)
        2. 从HDF5文件加载3D CT、PET和Lesion_mask数据
          3. 通过切片操作一次性提取三个连续切片 [slice_idx-1:slice_idx+2]
          4. 构建模态字典并交由transform执行归一化、数据增强、ToTensor等步骤
        
        Args:
            idx (int): 样本索引
        
        Returns:
            tuple[Dict[str, torch.Tensor], torch.Tensor]: 
                - modalities_dict: 包含三个模态的字典
                    - 'ct': 形状为 (3, H, W) 的CT 2.5D数据（若transform为默认设置则已归一化至[0, 1]）
                    - 'pet': 形状为 (3, H, W) 的PET 2.5D数据（同上）
                    - 'mask': 形状为 (3, H, W) 的Lesion_mask 2.5D数据
                - label: 对应的数字标签
        
        Raises:
            FileNotFoundError: 当HDF5文件不存在时
            KeyError: 当HDF5文件中缺少必要的数据集时
        """
        # --- 1. 获取样本信息 ---
        subject_id, slice_idx, label = self.samples[idx]
        
        # --- 2. 加载3D数据 ---
        # 构建HDF5文件路径
        h5_filename = f"{subject_id}.h5"
        h5_path = os.path.join(self.data_dir, h5_filename)
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5文件不存在: {h5_path}")
        
        # 定义模态配置：h5文件中的键名 -> 输出字典的键名
        modality_config = {
            'CT': 'ct',
            'PET': 'pet',
            'Lesion_mask': 'mask'
        }
        
        # 使用上下文管理器安全地打开HDF5文件，读取所有模态数据
        modality_volumes = {}
        with h5py.File(h5_path, 'r') as f:
            # 检查必要的数据集是否存在
            missing_keys = [key for key in modality_config.keys() if key not in f]
            if missing_keys:
                raise KeyError(f"HDF5文件 {h5_path} 缺少数据集: {missing_keys}")
            
            # 读取所有模态的3D数据 - 维度顺序: (H, W, D)
            for h5_key in modality_config.keys():
                modality_volumes[h5_key] = f[h5_key][:]  # shape: (H, W, D)
        
        # --- 3. 提取2.5D切片 ---
        # 获取深度维度的大小
        depth = modality_volumes['CT'].shape[2]  # D
        
        # 3.1检查边界条件
        if slice_idx <= 0 or slice_idx >= depth-1:
            raise IndexError(f"Subject {subject_id} 的 slice_idx {slice_idx} 越界，必须在 [1, {depth-2}] 之间以确保前后切片存在")
        
        # 3.2提取切片
        modality_slices = {}
        for h5_key in modality_config.keys():
            slices = modality_volumes[h5_key][:, :, slice_idx-1:slice_idx+2]  # shape: (H, W, 3)
            modality_slices[h5_key] = slices
        
        # --- 4. 构建样本字典并交给transform处理 ---
        modalities_raw: Dict[str, np.ndarray] = {}
        for h5_key, output_key in modality_config.items():
            # 保持 (H, W, 3) 格式，后续交由transform处理
            modalities_raw[output_key] = modality_slices[h5_key].astype(np.float32)

        sample = {
            'modalities': modalities_raw,
            'label': label
        }

        # --- 5. 应用transform（归一化、增强、ToTensor等） ---
        if self.transform is not None:
            transformed = self.transform(sample)
            modalities_tensor = transformed['modalities']
            label_tensor = transformed['label']
        else:
            # 回退逻辑：仅转换为Tensor (C, H, W)，不做归一化
            modalities_tensor = {
                key: torch.from_numpy(array.transpose(2, 0, 1)).float()
                for key, array in modalities_raw.items()
            }
            label_tensor = torch.tensor(label, dtype=torch.long)

        return modalities_tensor, label_tensor
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        获取数据集中各类别的样本数量分布。
        
        Returns:
            Dict[int, int]: 类别到样本数的映射字典
        """
        distribution = {}
        for _, _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def get_subject_ids(self) -> List[str]:
        """
        获取数据集中所有受试者的ID列表。
        
        Returns:
            List[str]: 受试者ID列表
        """
        return self.subject_ids.copy()
    
    def get_task_info(self) -> Dict:
        """
        获取当前任务的配置信息。
        
        Returns:
            Dict: 任务配置字典，包含description和labels
        """
        return self.task_config.copy()


def create_dataloaders(
    data_dir: str,
    participants_path: str,
    tasks_path: str,
    splits_path: str,
    lesion_info_path: str,
    task_name: str,
    fold: int,
    batch_size: int = 16,
    num_workers: int = 4,
    train_augmentations: Optional[Callable] = None,
    val_augmentations: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    便捷函数: 创建训练和验证数据加载器。
    
    Args:
        data_dir (str): HDF5数据目录
        participants_path (str): 参与者信息文件路径
        tasks_path (str): 任务配置文件路径
        splits_path (str): 交叉验证划分文件路径
        lesion_info_path (str): 病灶切片信息文件路径
        task_name (str): 任务名称
        fold (int): 交叉验证折数
        batch_size (int): 批次大小
        num_workers (int): 数据加载的工作进程数
    train_augmentations (callable, optional): 训练集数据增强（与albumentations的Compose配合使用，自动注入默认transform）
    val_augmentations (callable, optional): 验证集数据增强
    train_transform (callable, optional): 训练集自定义transform（优先于train_augmentations）
    val_transform (callable, optional): 验证集自定义transform
    
    Returns:
        Tuple[DataLoader, DataLoader]: 训练和验证数据加载器
    """
    # 若未提供transform，则根据augmentations构建默认transform
    if train_transform is None:
        train_transform = LungCancerTransform(augmentations=train_augmentations)
    if val_transform is None:
        val_transform = LungCancerTransform(augmentations=val_augmentations)

    # 创建训练数据集
    train_dataset = LungCancerDataset(
        data_dir=data_dir,
        participants_path=participants_path,
        tasks_path=tasks_path,
        splits_path=splits_path,
        lesion_info_path=lesion_info_path,
        task_name=task_name,
        fold=fold,
        mode='train',
        transform=train_transform
    )
    
    # 创建验证数据集
    val_dataset = LungCancerDataset(
        data_dir=data_dir,
        participants_path=participants_path,
        tasks_path=tasks_path,
        splits_path=splits_path,
        lesion_info_path=lesion_info_path,
        task_name=task_name,
        fold=fold,
        mode='val',
        transform=val_transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 使用示例
if __name__ == "__main__":
    """
    使用示例: 展示如何使用LungCancerDataset类
    """
    # 定义路径
    data_dir = "data/2_final_h5/"
    participants_path = "metadata/participants.tsv"
    tasks_path = "metadata/tasks.json"
    splits_path = "metadata/splits.json"
    lesion_info_path = "metadata/lesion_slice_info.json"
    
    # 构建默认transform（包含归一化、ToTensor，可选数据增强）
    default_transform = LungCancerTransform()

    # 创建数据集
    dataset = LungCancerDataset(
        data_dir=data_dir,
        participants_path=participants_path,
        tasks_path=tasks_path,
        splits_path=splits_path,
        lesion_info_path=lesion_info_path,
        task_name="3-class_classification",
        fold=0,
        mode='train',
        transform=default_transform
    )
    
    # 打印数据集信息
    print(f"\n数据集大小: {len(dataset)}")
    print(f"类别分布: {dataset.get_class_distribution()}")
    print(f"任务信息: {dataset.get_task_info()}")
    
    # 获取一个样本
    if len(dataset) > 0:
        modalities, sample_label = dataset[1000]
        print(f"\n样本模态:")
        print(f"  - CT形状: {modalities['ct'].shape}")      # 应该是 (3, H, W)
        print(f"  - PET形状: {modalities['pet'].shape}")    # 应该是 (3, H, W)
        print(f"  - Mask形状: {modalities['mask'].shape}")  # 应该是 (3, H, W)
        print(f"样本标签: {sample_label}")
        print(f"CT值范围: [{modalities['ct'].min():.3f}, {modalities['ct'].max():.3f}]")
        print(f"PET值范围: [{modalities['pet'].min():.3f}, {modalities['pet'].max():.3f}]")
        print(f"Mask值范围: [{modalities['mask'].min():.3f}, {modalities['mask'].max():.3f}]")
