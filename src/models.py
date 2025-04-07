"""
CNN模型定义模块

本模块包含用于图像分类的卷积神经网络模型:
- SimpleCNN: 简单的卷积神经网络，适用于MNIST和Fashion-MNIST数据集
- DeepCNN: 更深层的卷积神经网络，提供更好的特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class AttentionBlock(nn.Module):
    """注意力模块"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    """残差模块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)

        # 激活函数
        if activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = out * self.attention(out)

        out += self.shortcut(identity)
        out = self.activation(out)

        return out


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

    def __init__(self, config: ModelConfig) -> None:
        """初始化SimpleCNN模型的所有层"""
        super().__init__()
        self.config = config

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 残差块
        self.res1 = ResidualBlock(
            32, 64, use_attention=config.use_attention, activation=config.activation
        )

        # 池化和Dropout层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        # 全连接层 - 输入维度为64*14*14=12544
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

        # 激活函数
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "silu":
            self.activation = F.silu
        elif config.activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {config.activation}")

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
        x = self.activation(x)

        # 残差块
        x = self.res1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # 展平操作
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class DeepCNN(nn.Module):
    """
    更深层的卷积神经网络模型，提供更好的特征提取能力

    架构:
    - 多个残差块
    - 批量归一化
    - 注意力机制
    - Dropout层

    特征图尺寸变化:
    - 输入: [batch, 1, 28, 28]
    - 第一次池化后: [batch, 32, 14, 14]
    - 第二次池化后: [batch, 64, 7, 7]
    - 第三次池化后: [batch, 128, 3, 3]
    - 展平后: [batch, 128*3*3=1152]
    """

    def __init__(self, config: ModelConfig) -> None:
        """初始化DeepCNN模型的所有层"""
        super().__init__()
        self.config = config

        # 初始卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 残差块
        self.res1 = ResidualBlock(
            32, 64, use_attention=config.use_attention, activation=config.activation
        )
        self.res2 = ResidualBlock(
            64,
            128,
            stride=2,
            use_attention=config.use_attention,
            activation=config.activation,
        )
        self.res3 = ResidualBlock(
            128, 128, use_attention=config.use_attention, activation=config.activation
        )

        # 池化和Dropout层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(config.dropout_rate)
        self.dropout2 = nn.Dropout2d(config.dropout_rate)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, config.num_classes)

        # 激活函数
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "silu":
            self.activation = F.silu
        elif config.activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {config.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x: 输入图像张量，形状为[batch, 1, 28, 28]

        Returns:
            分类结果的对数概率
        """
        # 初始卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # 残差块
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # 展平和全连接操作
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_model(config: ModelConfig) -> nn.Module:
    """
    获取指定配置的模型实例

    Args:
        config: 模型配置

    Returns:
        对应的模型实例

    Raises:
        ValueError: 当指定的模型名称不存在时
    """
    if config.name.lower() == "simple":
        return SimpleCNN(config)
    elif config.name.lower() == "deep":
        return DeepCNN(config)
    else:
        raise ValueError(f"未知的模型名称: {config.name}")
