"""
数据加载模块

本模块提供加载和预处理数据集的功能:
- MNIST数据集
- Fashion-MNIST数据集
"""

from typing import Tuple, List
import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 标准化参数
DATASET_STATS = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,)},
    "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
}


def get_transform(dataset_name: str, augment: bool = False) -> transforms.Compose:
    """
    获取数据集的预处理变换

    Args:
        dataset_name: 数据集名称，'mnist'或'fashion_mnist'
        augment: 是否使用数据增强

    Returns:
        变换组合对象
    """
    # 确保数据集名称存在于预定义的统计信息中
    if dataset_name.lower() not in DATASET_STATS:
        raise ValueError(f"未知数据集: {dataset_name}")

    # 获取标准化参数
    stats = DATASET_STATS[dataset_name.lower()]

    if augment:
        # 使用数据增强
        return transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"]),
            ]
        )
    else:
        # 基本变换
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
        )


def get_mnist_loaders(
    batch_size: int = 64,
    test_batch_size: int = 1000,
    use_cuda: bool = False,
    val_ratio: float = 0.2,
    augment: bool = False,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载MNIST数据集并返回训练、验证和测试数据加载器

    Args:
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
        use_cuda: 是否使用CUDA
        val_ratio: 验证集比例
        augment: 是否使用数据增强
        data_dir: 数据存储目录

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 数据加载器的参数
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # 获取变换
    transform = get_transform("mnist", augment)

    # 下载MNIST数据集
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
    )

    # 测试集
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(
    batch_size: int = 64,
    test_batch_size: int = 1000,
    use_cuda: bool = False,
    val_ratio: float = 0.2,
    augment: bool = False,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载Fashion-MNIST数据集并返回训练、验证和测试数据加载器

    Args:
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
        use_cuda: 是否使用CUDA
        val_ratio: 验证集比例
        augment: 是否使用数据增强
        data_dir: 数据存储目录

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器
    """
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 数据加载器的参数
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # 获取变换
    transform = get_transform("fashion_mnist", augment)

    # 下载Fashion-MNIST数据集
    dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )

    # 划分训练集和验证集
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定随机种子
    )

    # 测试集
    test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
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


def get_data_loaders(
    dataset_name: str,
    batch_size: int = 64,
    test_batch_size: int = 1000,
    use_cuda: bool = False,
    val_ratio: float = 0.2,
    augment: bool = False,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    统一接口获取指定数据集的数据加载器

    Args:
        dataset_name: 数据集名称，'mnist'或'fashion_mnist'
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
        use_cuda: 是否使用CUDA
        val_ratio: 验证集比例
        augment: 是否使用数据增强
        data_dir: 数据存储目录

    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试数据加载器

    Raises:
        ValueError: 当指定的数据集名称不存在时
    """
    if dataset_name.lower() == "mnist":
        return get_mnist_loaders(
            batch_size, test_batch_size, use_cuda, val_ratio, augment, data_dir
        )
    elif dataset_name.lower() == "fashion_mnist":
        return get_fashion_mnist_loaders(
            batch_size, test_batch_size, use_cuda, val_ratio, augment, data_dir
        )
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
