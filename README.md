# 非侵入性肺癌亚型分类研究

本项目旨在使用深度学习技术，通过FDG-PET/CT影像对肺癌亚型（主要包括腺癌ADC、鳞状细胞癌SCC、小细胞肺癌SCLC）进行非侵入性分类。

## 项目结构

```
.
├── configs/                # 配置文件
│   └── main_config.yaml    # 主配置文件 (路径, 超参数等)
├── data/                   # 数据目录 (git忽略)
│   ├── 0_nifti/            # 阶段0: 原始NIfTI数据
│   ├── 1_preprocessed/     # 阶段 1: 空间预处理后数据
│   └── 2_final_h5/         # 阶段 2: 最终用于训练的H5数据
├── metadata/               # 元数据与划分信息
│   ├── subjects_master.csv # 病例主信息表 (标签, SUV ratio等)
│   └── splits.json         # 交叉验证划分
├── notebooks/              # Jupyter Notebooks
│   ├── 1_exploration/      # 探索性分析与代码测试
│   └── 2_pipelines/        # 流程编排与结果可视化
├── outputs/                # 实验输出 (git忽略)
│   ├── data_statistics/    # 数据统计结果
│   └── experiments/        # 每次训练的独立结果文件夹
├── scripts/                # 自动化脚本
│   ├── preprocess.py       # 一键执行完整预处理流程
│   └── train.py            # 启动模型训练
├── src/                    # 核心源代码模块
│   ├── dataset.py          # PyTorch Dataset类
│   ├── model.py            # 模型架构
│   ├── preprocessing.py    # 预处理功能函数
│   ├── engine.py           # 训练/验证循环
│   └── utils.py            # 通用辅助函数
├── environment.yml         # Conda环境配置
└── README.md               # 本项目说明
```

## 工作流程

### 1. 环境设置

使用Conda创建并激活项目环境：
```bash
conda env create -f environment.yml
conda activate lung_cancer_subtype
```

### 2. 数据准备

1.  将原始的NIfTI格式数据放置在 `data/0_nifti/` 目录下。
2.  确保 `metadata/subjects_master.csv` 文件包含所有病例的ID、病理标签、SUV ratio等必要信息。
3.  在 `configs/main_config.yaml` 中配置好所有数据路径和预处理参数。

### 3. 数据预处理

预处理流程被设计为可一键执行的脚本，它将按顺序生成`data/`目录下的中间和最终结果。

```bash
python scripts/preprocess.py --config configs/main_config.yaml
```
这个脚本会调用 `src/preprocessing.py` 中的函数，依次完成：
- 肺部分割后处理
- 空间重采样与对齐
- 基于肺部区域的裁切
- CT值的z-score归一化
- PET值的SUV转换与归一化
- 将最终数据保存为H5格式至 `data/2_final_h5/`

**或者**，你也可以在 `notebooks/2_pipelines/run_preprocessing_pipeline.ipynb` 中分步执行和检查预处理流程。

### 4. 模型训练

模型训练通过 `scripts/train.py` 脚本启动。

```bash
python scripts/train.py --config configs/main_config.yaml --exp_name "ResNet3D_baseline"
```

* `--exp_name`: (可选) 为本次实验指定一个名称。
* 脚本会自动在 `outputs/experiments/` 目录下创建一个以时间戳和实验名命名的文件夹，用于存放本次训练的所有结果（日志、模型权重、配置快照）。

### 5. 探索与分析

* **代码探索与调试**: 在 `notebooks/1_exploration/` 目录下的Notebooks中进行。
* **结果可视化与分析**: 训练完成后，在 `notebooks/2_pipelines/` 目录下的Notebooks中加载模型和结果，进行分析和绘图。

## 核心模块说明

-   **`src/preprocessing.py`**: 包含了所有数据预处理的核心算法函数，可被其他脚本或Notebook调用。
-   **`src/dataset.py`**: 定义了`LungCancerDataset`类，负责从`data/2_final_h5/`目录加载数据。
-   **`src/engine.py`**: 封装了标准的训练和验证循环逻辑，使得`train.py`脚本保持简洁。
-   **`configs/main_config.yaml`**: **唯一的参数配置入口**。所有的实验参数，如学习率、batch size、网络结构、数据路径等，都在此定义。

## 实验追踪

每次运行`scripts/train.py`都会在`outputs/experiments/`下生成一个唯一的实验目录，结构如下：

```
└── 20231027_1530_ResNet3D_baseline/
    ├── checkpoints/
    │   └── best_model.pth
    ├── logs.txt
    ├── config_snapshot.yaml
    └── tensorboard_logs/
```

这种方式确保了每一份实验结果都与其对应的配置和代码状态相关联，保证了研究的可复现性。

python scripts/train.py --task_name "ADC_vs_SCC" --fold 0 --lr 1e-5 --output_dir outputs/experiments/ADC_vs_SCC/fold_0