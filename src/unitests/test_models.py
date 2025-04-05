"""
测试模型模块
"""

import torch
from models import SimpleCNN, DeepCNN


def test_simple_cnn_init():
    """测试SimpleCNN初始化"""
    model = SimpleCNN()
    assert model is not None
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")
    assert hasattr(model, "dropout1")
    assert hasattr(model, "dropout2")
    assert hasattr(model, "pool")


def test_simple_cnn_forward():
    """测试SimpleCNN前向传播"""
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)


def test_deep_cnn_init():
    """测试DeepCNN初始化"""
    model = DeepCNN()
    assert model is not None
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")
    assert hasattr(model, "conv3")
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")
    assert hasattr(model, "dropout1")
    assert hasattr(model, "dropout2")
    assert hasattr(model, "pool")


def test_deep_cnn_forward():
    """测试DeepCNN前向传播"""
    model = DeepCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
