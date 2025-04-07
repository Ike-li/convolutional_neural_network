"""
数据加载模块

本模块提供加载和预处理数据集的功能:
- MNIST数据集
- Fashion-MNIST数据集
"""

from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

from .config import Config, DataConfig


class CachedDataset(Dataset):
    """带缓存的数据集"""

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "dataset_cache.pkl"
        self.cached_data = self._load_cache()

    def _load_cache(self) -> Dict[int, Any]:
        """加载缓存数据"""
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self) -> None:
        """保存缓存数据"""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cached_data, f)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self.cached_data:
            img, target = self.cached_data[idx]
            if self.transform:
                img = self.transform(img)
            return img, target

        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)

        # 缓存转换后的数据
        self.cached_data[idx] = (img, target)
        if len(self.cached_data) % 1000 == 0:  # 每1000个样本保存一次缓存
            self._save_cache()

        return img, target

    def __len__(self) -> int:
        return len(self.dataset)


# 标准化参数
DATASET_STATS = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,)},
    "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
}


def get_transform(config: DataConfig) -> transforms.Compose:
    """
    获取数据集的预处理变换

    Args:
        config: 数据配置

    Returns:
        变换组合对象
    """
    # 确保数据集名称存在于预定义的统计信息中
    if config.dataset.lower() not in DATASET_STATS:
        raise ValueError(f"未知数据集: {config.dataset}")

    # 获取标准化参数
    stats = DATASET_STATS[config.dataset.lower()]

    if config.augment:
        # 使用数据增强
        return transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomErasing(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    else:
        # 基本变换
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
        )


def get_mnist_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载MNIST数据集并返回训练、验证和测试数据加载器

    Args:
        config: 配置对象

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 创建数据目录
    data_dir = Path(config.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载器的参数
    kwargs = {
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
    }

    # 获取变换
    transform = get_transform(config.data)

    # 下载MNIST数据集
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int((1 - config.data.val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
    )

    # 测试集
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    # 使用缓存数据集
    if config.data.cache_data:
        cache_dir = data_dir / "cache" / "mnist"
        train_dataset = CachedDataset(train_dataset, cache_dir, transform)
        val_dataset = CachedDataset(val_dataset, cache_dir, transform)
        test_dataset = CachedDataset(test_dataset, cache_dir, transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.test_batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载Fashion-MNIST数据集并返回训练、验证和测试数据加载器

    Args:
        config: 配置对象

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 创建数据目录
    data_dir = Path(config.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载器的参数
    kwargs = {
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
    }

    # 获取变换
    transform = get_transform(config.data)

    # 下载Fashion-MNIST数据集
    dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )

    # 划分训练集和验证集
    train_size = int((1 - config.data.val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
    )

    # 测试集
    test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)

    # 使用缓存数据集
    if config.data.cache_data:
        cache_dir = data_dir / "cache" / "fashion_mnist"
        train_dataset = CachedDataset(train_dataset, cache_dir, transform)
        val_dataset = CachedDataset(val_dataset, cache_dir, transform)
        test_dataset = CachedDataset(test_dataset, cache_dir, transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.test_batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, val_loader, test_loader


def get_dataset_classes(dataset_name: str) -> List[str]:
    """
    获取数据集的类别名称

    Args:
        dataset_name: 数据集名称，'mnist'或'fashion_mnist'

    Returns:
        类别名称列表

    Raises:
        ValueError: 当指定的数据集名称不存在时
    """
    if dataset_name.lower() == "mnist":
        return [str(i) for i in range(10)]
    elif dataset_name.lower() == "fashion_mnist":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    根据配置获取数据加载器

    Args:
        config: 配置对象

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    if config.data.dataset.lower() == "mnist":
        return get_mnist_loaders(config)
    elif config.data.dataset.lower() == "fashion_mnist":
        return get_fashion_mnist_loaders(config)
    else:
        raise ValueError(f"不支持的数据集: {config.data.dataset}")
