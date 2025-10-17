# src/utils.py

import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def set_seed(seed: int = 42):
    """设置随机种子以确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保每次运行的卷积算法是固定的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AverageMeter:
    """计算并存储平均值和当前值。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_metrics(y_true, y_pred, y_prob=None, num_classes=3):
    """
    计算分类任务的核心指标。
    Args:
        y_true: 真实标签 (1D array)
        y_pred: 预测标签 (1D array)
        y_prob: 预测概率 (2D array, shape: [n_samples, n_classes])
    Returns:
        一个包含各项指标的字典。
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    if y_prob is not None and num_classes > 1:
        if num_classes == 2:
            # 二分类AUC
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # 多分类AUC (One-vs-Rest)
            metrics['auc_ovr_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
            
    # 计算混淆矩阵并转换为列表以便于json序列化
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics