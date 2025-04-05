"""
测试训练器模块
"""

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from trainer import Trainer
from models import SimpleCNN


@pytest.fixture
def dummy_data():
    """创建用于测试的虚拟数据集"""
    # 创建一些随机数据
    x_train = torch.randn(100, 1, 28, 28)
    y_train = torch.randint(0, 10, (100,))
    x_val = torch.randn(20, 1, 28, 28)
    y_val = torch.randint(0, 10, (20,))
    x_test = torch.randn(20, 1, 28, 28)
    y_test = torch.randint(0, 10, (20,))

    # 创建DataLoader
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=16, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=8, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=8, shuffle=False)

    return train_loader, val_loader, test_loader


def test_trainer_init(dummy_data):
    """测试Trainer类初始化"""
    train_loader, val_loader, test_loader = dummy_data
    model = SimpleCNN()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=0.01,
        momentum=0.5,
        optimizer_name="sgd",
    )

    assert trainer.model is model
    assert trainer.train_loader is train_loader
    assert trainer.val_loader is val_loader
    assert trainer.test_loader is test_loader
    assert trainer.device is device
    assert isinstance(trainer.optimizer, optim.SGD)
    # 修改测试，Trainer实际上并没有使用loss_fn属性
    assert hasattr(trainer, "train_losses")
    # Trainer也没有history属性，使用train_losses和train_accs
    assert hasattr(trainer, "train_accs")


def test_trainer_train_epoch(dummy_data):
    """测试Trainer类的训练Epoch方法"""
    train_loader, val_loader, test_loader = dummy_data
    model = SimpleCNN()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=0.01,
        momentum=0.5,
        optimizer_name="sgd",
    )

    # 训练一个epoch，使用正确的方法名
    loss, acc = trainer.train_epoch(epoch=1)

    # 验证返回值
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100


def test_trainer_validate(dummy_data):
    """测试Trainer类的validate方法"""
    train_loader, val_loader, test_loader = dummy_data
    model = SimpleCNN()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=0.01,
        momentum=0.5,
        optimizer_name="sgd",
    )

    # 执行验证
    loss, acc = trainer.validate()

    # 验证返回值
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100


def test_trainer_test(dummy_data):
    """测试Trainer类的test方法"""
    train_loader, val_loader, test_loader = dummy_data
    model = SimpleCNN()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=0.01,
        momentum=0.5,
        optimizer_name="sgd",
    )

    # 执行测试
    loss, acc = trainer.test()

    # 验证返回值
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100
