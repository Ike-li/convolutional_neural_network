"""
CNN项目

一个基于PyTorch的卷积神经网络项目，用于图像分类任务
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import ModelFactory
from .data.datasets import DatasetFactory
from .training.trainer import Trainer
from .utils import Visualizer, Logger

__all__ = [
    "ModelFactory",
    "DatasetFactory",
    "Trainer",
    "Visualizer",
    "Logger",
]
