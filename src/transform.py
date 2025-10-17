# src/transforms.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

from monai.transforms import (
    Compose,
    Transposed,
    EnsureChannelFirstd,
    RandSpatialCropd,
    Resized,
    SpatialPadd,
    RandRotated,
    RandFlipd,
    ToTensord
)

# 此处为单独构建的处理流程
class IntensityNormalized:
    """
    输入: data_dict (字典), 其中包含待处理的NumPy数组。
    输出: data_dict (字典), 其中指定的键值已被归一化。
    归一化公式: (x - min) / (max - min)，并裁剪到[0, 1]范围内。
    """
    def __init__(self, keys: List[str], normalization_ranges: Dict[str, Tuple[float, float]]):
        self.keys = keys
        self.ranges = normalization_ranges

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d = dict(data)  # 复制字典，避免修改原始数据
        for key in self.keys:
            if key in d and key in self.ranges:
                min_val, max_val = self.ranges[key]
                # 确保数据类型为float，以进行浮点数运算
                d[key] = d[key].astype(np.float32)
                d[key] = np.clip(d[key], min_val, max_val)
                d[key] = (d[key] - min_val) / (max_val - min_val)
        return d


# 利用Compose将MONAI提供的变化和另外编写的变换进行组合；对val和train进行区分，将超参数作为输入传递
def get_transforms(
    mode: str,
    normalization_ranges: Dict[str, Tuple[float, float]] = {
        'ct': (-1024.0, 600.0),
        'pet': (0.0, 30.0)
    },
    output_size: Tuple[int, int] = (224, 224),
) -> Callable:
    """
    根据模式（‘train’或者‘val’）构建并返回一个复合变换函数，用于对输入数据进行一系列预处理操作。
    
    Args:
        mode (str): 变换模式，'train'表示训练模式，'val'表示验证模式。
        normalization_ranges (Dict[str, Tuple[float, float]]): 每个模态的归一化范围。
        output_size (Tuple[int, int]): 目标输出尺寸 (H, W)。

    Returns:
        Callable: 完整的变换流水线。
    """
    # train和val都需要完成的transform
    # 包括数值归一化和确保通道维度在前
    base_transforms = [
        IntensityNormalized(keys=['ct', 'pet'], normalization_ranges=normalization_ranges),
        Transposed(keys=['ct', 'pet', 'mask'], indices=(2, 1, 0)),  # 将WHC转为CHW
    ]
    tensor_transforms = [ToTensord(keys=['ct', 'pet', 'mask', 'label'])]  # 最后转换为Tensor

    if mode == 'train':
        train_transforms = [
            SpatialPadd(keys=['ct', 'pet', 'mask'], spatial_size=(256, 256), mode='constant', constant_values=0),   # 先填充到256x256
            RandSpatialCropd(keys=['ct', 'pet', 'mask'], roi_size=output_size, random_size=False, random_center=True),  # 再随机裁剪到指定大小
        ]

        return Compose(base_transforms + train_transforms + tensor_transforms)
    
    elif mode == 'val':
        val_transforms = [
            SpatialPadd(keys=['ct', 'pet', 'mask'], spatial_size=output_size, mode='constant', constant_values=0)
        ]

        return Compose(base_transforms + val_transforms + tensor_transforms)
    
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'val'.")