"""
测试数据加载模块
"""

import pytest
import torch
from data_loader import (
    get_dataset_classes, 
    get_mnist_loaders, 
    get_fashion_mnist_loaders, 
    get_data_loaders
)


def test_get_dataset_classes():
    """测试获取数据集类别函数"""
    mnist_classes = get_dataset_classes("mnist")
    assert len(mnist_classes) == 10
    assert isinstance(mnist_classes, list)
    
    fashion_classes = get_dataset_classes("fashion_mnist")
    assert len(fashion_classes) == 10
    assert isinstance(fashion_classes, list)
    
    with pytest.raises(ValueError):
        get_dataset_classes("unknown_dataset")


def test_get_mnist_loaders():
    """测试获取MNIST数据加载器"""
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=64, 
        test_batch_size=100,
        val_ratio=0.1,
        augment=False
    )
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    batch = next(iter(train_loader))
    assert len(batch) == 2  # 数据和标签
    assert isinstance(batch[0], torch.Tensor)
    assert isinstance(batch[1], torch.Tensor)
    

def test_get_fashion_mnist_loaders():
    """测试获取Fashion-MNIST数据加载器"""
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=64, 
        test_batch_size=100,
        val_ratio=0.1,
        augment=False
    )
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    batch = next(iter(train_loader))
    assert len(batch) == 2  # 数据和标签
    assert isinstance(batch[0], torch.Tensor)
    assert isinstance(batch[1], torch.Tensor)


def test_get_data_loaders():
    """测试通用数据加载器函数"""
    # 测试MNIST
    loaders = get_data_loaders(
        dataset_name="mnist",
        batch_size=64,
        test_batch_size=100,
        val_ratio=0.1
    )
    assert len(loaders) == 3
    
    # 测试Fashion-MNIST
    loaders = get_data_loaders(
        dataset_name="fashion_mnist",
        batch_size=64,
        test_batch_size=100,
        val_ratio=0.1
    )
    assert len(loaders) == 3
    
    # 测试未知数据集
    with pytest.raises(ValueError):
        get_data_loaders(
            dataset_name="unknown_dataset",
            batch_size=64,
            test_batch_size=100
        ) 