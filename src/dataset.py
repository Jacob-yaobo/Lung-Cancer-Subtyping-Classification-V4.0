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

from src.transforms import get_transforms

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
        self.transform = transform
        
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

        if slice_idx <= 0 or slice_idx >= depth-1:
            raise IndexError(f"Subject {subject_id} 的 slice_idx {slice_idx} 越界，必须在 [1, {depth-2}] 之间以确保前后切片存在")
        
        # 3.1 为CT和PET提取前、中、后三层切片构建2.5D数据
        ct_slices = modality_volumes['CT'][:, :, slice_idx-1:slice_idx+2]  # shape: (H, W, 3)
        pet_slices = modality_volumes['PET'][:, :, slice_idx-1:slice_idx+2]  # shape: (H, W, 3) 

        # 3.2 Lesion_mask只需要提取中间切片，并扩展channel
        mask_slice_2d = modality_volumes['Lesion_mask'][:, :, slice_idx] # (W, H)
        mask_slice = np.expand_dims(mask_slice_2d, axis=-1)

        # --- 4. 构建样本字典并交给transform处理 ---
        sample = {
            'ct': ct_slices.astype(np.float32),
            'pet': pet_slices.astype(np.float32),
            'mask': mask_slice.astype(np.float32),
            'label': label
        }

        # --- 5. 应用transform（归一化、增强、ToTensor等） ---
        if self.transform:  
            transformed_sample = self.transform(sample)

        model_inputs = {
            'ct': transformed_sample['ct'],       # Tensor, shape: (3, H, W)
            'pet': transformed_sample['pet'],     # Tensor, shape: (3, H, W)
            'mask': transformed_sample['mask']    # Tensor, shape: (3, H, W)
        }

        label_tensor = transformed_sample['label']

        return model_inputs, label_tensor, subject_id
    

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


# ===================================================================
# 模块自测试代码
# ===================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 定义路径
    DATA_DIR = "data/2_final_h5/"
    PARTICIPANTS_PATH = "metadata/participants.tsv"
    TASKS_PATH = "metadata/tasks.json"
    SPLITS_PATH = "metadata/splits.json"
    LESION_INFO_PATH = "metadata/lesion_slice_info.json"

    print("--- Running LungCancerDataset self-test ---")

    # --- 2. 实例化数据集 ---
    print("\n[INFO] Initializing training dataset for fold 0...")
    train_dataset = LungCancerDataset(
        data_dir=DATA_DIR,
        participants_path=PARTICIPANTS_PATH,
        tasks_path=TASKS_PATH,
        splits_path=SPLITS_PATH,
        lesion_info_path=LESION_INFO_PATH,
        task_name="ADC_vs_SCC",
        fold=0,
        mode='train'
    )
    
    print("\n[INFO] Initializing validation dataset for fold 0...")
    val_dataset = LungCancerDataset(
        data_dir=DATA_DIR,
        participants_path=PARTICIPANTS_PATH,
        tasks_path=TASKS_PATH,
        splits_path=SPLITS_PATH,
        lesion_info_path=LESION_INFO_PATH,
        task_name="ADC_vs_SCC",
        fold=0,
        mode='val'
    )

    # --- 3. 检查数据集信息 ---
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train dataset class distribution: {train_dataset.get_class_distribution()}")
    print(f"Validation dataset class distribution: {val_dataset.get_class_distribution()}")

    # --- 4. 随机获取并检查一个训练样本 ---
    if len(train_dataset) > 0:
        print("\n--- Checking a random training sample ---")
        random_idx = np.random.randint(0, len(train_dataset))
        inputs, label = train_dataset[random_idx]

        print(f"Sample Index: {random_idx}")
        print(f"Inputs keys: {inputs.keys()}")
        print(f"CT shape: {inputs['ct'].shape}, dtype: {inputs['ct'].dtype}")
        print(f"PET shape: {inputs['pet'].shape}, dtype: {inputs['pet'].dtype}")
        print(f"Mask shape: {inputs['mask'].shape}, dtype: {inputs['mask'].dtype}")
        print(f"Label: {label}, dtype: {label.dtype}")
        
        # 检查数据范围
        print(f"CT value range: [{inputs['ct'].min():.2f}, {inputs['ct'].max():.2f}]")
        print(f"PET value range: [{inputs['pet'].min():.2f}, {inputs['pet'].max():.2f}]")
        
        # --- 5. 可视化样本 ---
        # 可视化对于检查数据增强是否正确应用至关重要
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Train Sample #{random_idx} - Label: {label.item()}")
        
        # CT (取中心切片，即第1个通道)
        axes[0].imshow(inputs['ct'][1, :, :], cmap='gray')
        axes[0].set_title("CT (Center Slice)")
        
        # PET (取中心切片)
        axes[1].imshow(inputs['pet'][1, :, :], cmap='hot')
        axes[1].set_title("PET (Center Slice)")

        # Mask (取中心切片)
        axes[2].imshow(inputs['mask'][1, :, :], cmap='jet')
        axes[2].set_title("Mask (Center Slice)")
        
        plt.tight_layout()
        plt.show()
