"""
配置文件

本模块包含模型训练和评估所需的所有超参数配置。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""

    name: str = "simple"  # 模型名称：'simple' 或 'deep'
    num_classes: int = 10  # 分类数量
    use_residual: bool = True  # 是否使用残差连接
    use_attention: bool = True  # 是否使用注意力机制
    activation: str = "relu"  # 激活函数：'relu', 'silu', 'gelu'
    dropout_rate: float = 0.5  # Dropout比率


@dataclass
class TrainingConfig:
    """训练配置"""

    batch_size: int = 64  # 训练批次大小
    test_batch_size: int = 1000  # 测试批次大小
    epochs: int = 50  # 训练轮数
    lr: float = 0.01  # 学习率
    momentum: float = 0.5  # 动量
    weight_decay: float = 1e-4  # 权重衰减
    optimizer: str = "adam"  # 优化器：'sgd', 'adam'
    scheduler: str = "plateau"  # 学习率调度器：'plateau', 'cosine', 'step', 'none'
    early_stopping_patience: int = 10  # 早停耐心值
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    mixed_precision: bool = True  # 是否使用混合精度训练
    log_interval: int = 100  # 日志打印间隔
    save_dir: str = "models"  # 模型保存目录


@dataclass
class DataConfig:
    """数据配置"""

    dataset: str = "mnist"  # 数据集名称：'mnist' 或 'fashion_mnist'
    data_dir: str = "data"  # 数据存储目录
    val_ratio: float = 0.2  # 验证集比例
    augment: bool = True  # 是否使用数据增强
    num_workers: int = 4  # 数据加载线程数
    pin_memory: bool = True  # 是否使用固定内存
    cache_data: bool = True  # 是否缓存数据


@dataclass
class Config:
    """总配置"""

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    seed: int = 42  # 随机种子
    device: str = "cuda"  # 训练设备：'cuda' 或 'cpu'
    debug: bool = False  # 是否启用调试模式
