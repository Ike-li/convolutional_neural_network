"""
配置模块

包含所有常量和默认设置
"""

from typing import Dict, Tuple

# 数据集配置
DATASET_STATS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,)},
    "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
}

# 数据加载配置
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_TEST_BATCH_SIZE: int = 1000
DEFAULT_VAL_RATIO: float = 0.2
DEFAULT_DATA_DIR: str = "data"

# 训练配置
DEFAULT_LEARNING_RATE: float = 0.01
DEFAULT_MOMENTUM: float = 0.5
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_OPTIMIZER: str = "sgd"
DEFAULT_SCHEDULER: str = "plateau"
DEFAULT_LOG_INTERVAL: int = 100
DEFAULT_SAVE_DIR: str = "models"
DEFAULT_EARLY_STOPPING_PATIENCE: int = 10

# 日志配置
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FILE: str = "training.log"

# 模型配置
SIMPLE_CNN_CONFIG = {
    "conv1": {
        "in_channels": 1,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv2": {
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "fc1": {"in_features": 64 * 14 * 14, "out_features": 128},
    "fc2": {"in_features": 128, "out_features": 10},
    "dropout1": 0.25,
    "dropout2": 0.5,
}

DEEP_CNN_CONFIG = {
    "conv1": {
        "in_channels": 1,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv2": {
        "in_channels": 32,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "conv3": {
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    "fc1": {"in_features": 128 * 3 * 3, "out_features": 256},
    "fc2": {"in_features": 256, "out_features": 10},
    "dropout1": 0.25,
    "dropout2": 0.5,
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    "rotation": 10,
    "translate": (0.1, 0.1),
    "scale": (0.9, 1.1),
    "brightness": 0.2,
    "contrast": 0.2,
}

# 日志配置
LOG_CONFIG = {
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "level": "INFO",
    "file": "training.log",
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 100,
    "style": "seaborn",
    "save_format": "png",
}

# 数据加载配置
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
NUM_WORKERS = 4
PIN_MEMORY = True

# 训练配置
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 10
LOG_INTERVAL = 10

# 模型配置
MODEL_CONFIGS = {
    "simple_cnn": {
        "input_channels": 1,
        "conv_channels": [32, 64],
        "conv_kernel_sizes": [3, 3],
        "conv_paddings": [1, 1],
        "pool_kernel_size": 2,
        "dropout_rate": 0.5,
        "fc_sizes": [12544, 128, 10],
    },
    "deep_cnn": {
        "input_channels": 1,
        "conv_channels": [32, 64, 128],
        "conv_kernel_sizes": [3, 3, 3],
        "conv_paddings": [1, 1, 1],
        "pool_kernel_size": 2,
        "dropout_rate": 0.5,
        "fc_sizes": [1152, 256, 10],
    },
}

# 可视化配置
DEFAULT_FIGURE_SIZE = (10, 6)  # 默认图形大小
DEFAULT_DPI = 100  # 默认DPI
DEFAULT_STYLE = "seaborn"  # 默认绘图风格
DEFAULT_SAVE_FORMAT = "png"  # 默认保存格式
PLOT_STYLE = "seaborn"  # 绘图风格
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # 颜色方案

# 路径配置
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"

# 其他配置
SEED = 42
USE_CUDA = True
