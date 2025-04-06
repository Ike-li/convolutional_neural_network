"""
训练器模块

提供模型训练和评估功能
"""

import os
import logging
import time
from typing import Tuple, List, Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..config.settings import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_SAVE_DIR,
    DEFAULT_EARLY_STOPPING_PATIENCE,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    模型训练器类

    用于训练、验证和测试神经网络模型，并记录训练过程中的指标。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        lr: float = DEFAULT_LEARNING_RATE,
        momentum: float = DEFAULT_MOMENTUM,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        optimizer_name: str = DEFAULT_OPTIMIZER,
        scheduler_name: str = DEFAULT_SCHEDULER,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        save_dir: str = DEFAULT_SAVE_DIR,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
        callbacks: Optional[List[Callable]] = None,
    ) -> None:
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 训练设备 (CPU/GPU)
            lr: 学习率
            momentum: 动量（SGD优化器）
            weight_decay: 权重衰减
            optimizer_name: 优化器名称 ("sgd", "adam")
            scheduler_name: 学习率调度器名称 ("plateau", "cosine", "step", "none")
            log_interval: 日志打印间隔
            save_dir: 模型保存目录
            early_stopping_patience: 早停耐心值，如果为0则禁用早停
            callbacks: 回调函数列表
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.early_stopping_patience = early_stopping_patience
        self.callbacks = callbacks or []

        # 创建优化器
        self.optimizer = self._create_optimizer(
            optimizer_name, lr, momentum, weight_decay
        )

        # 创建学习率调度器
        self.scheduler = self._create_scheduler(scheduler_name)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练记录
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []

        # 训练统计
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.start_time: float = 0.0
        self.early_stopping_counter: int = 0

        # 将模型移动到指定设备
        self.model.to(device)

    def _create_optimizer(
        self, optimizer_name: str, lr: float, momentum: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        创建优化器

        Args:
            optimizer_name: 优化器名称
            lr: 学习率
            momentum: 动量
            weight_decay: 权重衰减

        Returns:
            优化器实例

        Raises:
            ValueError: 当指定的优化器名称不存在时
        """
        if optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"未知的优化器: {optimizer_name}")

    def _create_scheduler(self, scheduler_name: str) -> Optional[Any]:
        """
        创建学习率调度器

        Args:
            scheduler_name: 调度器名称

        Returns:
            学习率调度器实例，如果scheduler_name为"none"则返回None

        Raises:
            ValueError: 当指定的调度器名称不存在时
        """
        if scheduler_name.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", patience=3, factor=0.1, verbose=True
            )
        elif scheduler_name.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=1e-6
            )
        elif scheduler_name.lower() == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name.lower() == "none":
            return None
        else:
            raise ValueError(f"未知的学习率调度器: {scheduler_name}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch数

        Returns:
            平均损失和准确率
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播和反向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # 累计统计
            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            # 更新进度条
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": train_loss / total,
                        "acc": 100.0 * correct / total,
                    }
                )

            # 执行回调
            for callback in self.callbacks:
                callback(
                    epoch=epoch,
                    batch=batch_idx,
                    loss=loss.item(),
                    acc=100.0 * correct / total,
                )

        # 计算平均损失和准确率
        train_loss /= len(self.train_loader.dataset)
        train_acc = 100.0 * correct / len(self.train_loader.dataset)

        # 保存训练记录
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        return train_loss, train_acc

    def validate(self) -> Tuple[float, float]:
        """
        在验证集上评估模型

        Returns:
            平均损失和准确率
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)
                val_loss += F.nll_loss(output, target, reduction="sum").item()

                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # 计算平均损失和准确率
        val_loss /= len(self.val_loader.dataset)
        val_acc = 100.0 * correct / len(self.val_loader.dataset)

        # 保存验证记录
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

        # 更新学习率（如果使用ReduceLROnPlateau调度器）
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)

        return val_loss, val_acc

    def test(self) -> Tuple[float, float]:
        """
        在测试集上评估模型

        Returns:
            平均损失和准确率
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()

                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # 计算平均损失和准确率
        test_loss /= len(self.test_loader.dataset)
        test_acc = 100.0 * correct / len(self.test_loader.dataset)

        return test_loss, test_acc

    def train(self, epochs: int) -> Tuple[float, str]:
        """
        训练模型

        Args:
            epochs: 训练轮数

        Returns:
            最佳验证准确率和模型保存路径
        """
        self.start_time = time.time()
        best_model_path = ""

        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 在验证集上评估
            val_loss, val_acc = self.validate()

            # 记录训练信息
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_model_path = os.path.join(
                    self.save_dir, f"best_model_epoch_{epoch}.pth"
                )
                self.model.save(best_model_path)
                logger.info(f"保存最佳模型到 {best_model_path}")

            # 早停检查
            if self.early_stopping_patience > 0:
                if val_acc > self.best_val_acc:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(
                            f"早停触发！{self.early_stopping_patience} 个epoch未改善"
                        )
                        break

        # 训练结束，加载最佳模型
        if best_model_path:
            self.model = self.model.load(best_model_path, self.device)

        return self.best_val_acc, best_model_path

    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要信息

        Returns:
            包含训练统计信息的字典
        """
        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": time.time() - self.start_time,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }
