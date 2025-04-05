"""
CNN模型定义模块

本模块包含用于图像分类的卷积神经网络模型:
- SimpleCNN: 简单的卷积神经网络，适用于MNIST和Fashion-MNIST数据集
- DeepCNN: 更深层的卷积神经网络，提供更好的特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络模型，适用于MNIST和Fashion-MNIST数据集

    架构:
    - 两个卷积层（带padding=1保持特征图尺寸）
    - 一个最大池化层（在第二次卷积后应用）
    - Dropout层
    - 两个全连接层

    特征图尺寸变化:
    - 输入: [batch, 1, 28, 28]
    - 第一次卷积: [batch, 32, 28, 28]
    - 第二次卷积: [batch, 64, 28, 28]
    - 池化后: [batch, 64, 14, 14]
    - 展平后: [batch, 64*14*14=12544]
    """

    def __init__(self) -> None:
        """初始化SimpleCNN模型的所有层"""
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 池化和Dropout层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # 全连接层 - 输入维度为64*14*14=12544
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入图像张量，形状为[batch, 1, 28, 28]

        Returns:
            分类结果的对数概率
        """
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        # 展平操作
        x = torch.flatten(x, 1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class DeepCNN(nn.Module):
    """
    更深层的卷积神经网络模型，提供更好的特征提取能力

    架构:
    - 三个卷积层，每个卷积层后接批量归一化
    - 三个最大池化层（每个卷积层后）
    - Dropout层
    - 两个全连接层

    特征图尺寸变化:
    - 输入: [batch, 1, 28, 28]
    - 第一次池化后: [batch, 32, 14, 14]
    - 第二次池化后: [batch, 64, 7, 7]
    - 第三次池化后: [batch, 128, 3, 3]
    - 展平后: [batch, 128*3*3=1152]
    """

    def __init__(self) -> None:
        """初始化DeepCNN模型的所有层"""
        super().__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化和Dropout层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # 全连接层 - 输入维度为128*3*3=1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入图像张量，形状为[batch, 1, 28, 28]

        Returns:
            分类结果的对数概率
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # 展平和全连接操作
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    获取指定名称的模型实例

    Args:
        model_name: 模型名称，'simple'或'deep'
        num_classes: 分类数量，默认为10

    Returns:
        对应的模型实例

    Raises:
        ValueError: 当指定的模型名称不存在时
    """
    if model_name.lower() == "simple":
        return SimpleCNN()
    elif model_name.lower() == "deep":
        return DeepCNN()
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
