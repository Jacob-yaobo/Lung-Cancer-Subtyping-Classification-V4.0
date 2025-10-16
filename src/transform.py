class LungCancerTransform:
    """预处理和增强变换，用于标准化模态数据并转换为Tensor。

    功能:
    - 根据预设范围对不同模态执行线性归一化
    - 可选地应用一致的数据增强（如albumentations）
    - 将数组从(H, W, C)转为(C, H, W)并转换为PyTorch Tensor
    """

    def __init__(
        self,
        augmentations: Optional[Callable] = None,
        normalization_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        to_tensor: bool = True,
    ) -> None:
        self.augmentations = augmentations
        # 默认归一化范围：CT使用HU值范围，PET使用SUV范围
        default_ranges: Dict[str, Tuple[float, float]] = {
            'ct': (-1024.0, 600.0),
            'pet': (0.0, 30.0),
        }
        if normalization_ranges is not None:
            default_ranges.update(normalization_ranges)
        self.normalization_ranges = default_ranges
        self.to_tensor = to_tensor

    def _normalize(self, data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        data = np.clip(data, min_val, max_val)
        return (data - min_val) / (max_val - min_val)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        modalities: Dict[str, np.ndarray] = sample['modalities']
        label = sample['label']

        # 拷贝并转换为float32，确保后续处理安全
        arrays: Dict[str, np.ndarray] = {
            key: value.astype(np.float32, copy=True)
            for key, value in modalities.items()
        }

        # 归一化（如果配置中包含该模态）
        for modality, (min_val, max_val) in self.normalization_ranges.items():
            if modality in arrays:
                arrays[modality] = self._normalize(arrays[modality], min_val, max_val)

        # 应用数据增强。默认使用CT作为主图像键
        if self.augmentations is not None:
            if 'ct' in arrays:
                main_key = 'ct'
            else:
                # 若不存在ct，则以迭代顺序的第一个模态作为主键
                main_key = next(iter(arrays))

            aug_inputs: Dict[str, np.ndarray] = {'image': arrays[main_key]}
            for key, array in arrays.items():
                if key == main_key:
                    continue
                aug_inputs[key] = array

            augmented = self.augmentations(**aug_inputs)
            arrays[main_key] = augmented['image']
            for key in arrays.keys():
                if key == main_key:
                    continue
                if key in augmented:
                    arrays[key] = augmented[key]

        # 转换为Tensor格式 (C, H, W)
        if self.to_tensor:
            tensor_modalities: Dict[str, torch.Tensor] = {}
            for key, array in arrays.items():
                if array.ndim != 3:
                    raise ValueError(f"模态 '{key}' 的数组维度为 {array.ndim}，期望为3")
                chw = np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
                tensor_modalities[key] = torch.from_numpy(chw)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return {
                'modalities': tensor_modalities,
                'label': label_tensor
            }

        # 若不转换为Tensor，则仍以NumPy数组返回
        return {
            'modalities': arrays,
            'label': label
        }
