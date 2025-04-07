"""
模型训练器模块

本模块提供用于训练和评估深度学习模型的Trainer类。
"""

import logging
import time
from typing import Tuple, List, Dict, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config, TrainingConfig

logger = logging.getLogger("CNN.trainer")


class EarlyStopping:
    """早停机制"""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, score: float, model: nn.Module, path: str) -> bool:
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0
        return False

    def save_checkpoint(self, score: float, model: nn.Module, path: str) -> None:
        """保存检查点"""
        if self.verbose:
            logger.info(
                f"Validation score improved ({self.val_loss_min:.6f} --> {score:.6f})"
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = score


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
        config: Config,
    ) -> None:
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # 创建优化器
        self.optimizer = self._create_optimizer(config.training)

        # 创建学习率调度器
        self.scheduler = self._create_scheduler(config.training)

        # 创建早停机制
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience, verbose=True
        )

        # 创建保存目录
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 训练记录
        self.train_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_losses: List[float] = []
        self.val_accs: List[float] = []

        # 训练统计
        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0
        self.start_time: float = 0.0

        # 混合精度训练
        self.scaler = GradScaler() if config.training.mixed_precision else None

    def _create_optimizer(self, config: TrainingConfig) -> torch.optim.Optimizer:
        """
        创建优化器

        Args:
            config: 训练配置

        Returns:
            优化器实例

        Raises:
            ValueError: 当指定的优化器名称不存在时
        """
        if config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"未知的优化器: {config.optimizer}")

    def _create_scheduler(self, config: TrainingConfig) -> Optional[Any]:
        """
        创建学习率调度器

        Args:
            config: 训练配置

        Returns:
            学习率调度器实例，如果scheduler_name为"none"则返回None

        Raises:
            ValueError: 当指定的调度器名称不存在时
        """
        if config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", patience=3, factor=0.1, verbose=True
            )
        elif config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=1e-6
            )
        elif config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif config.scheduler.lower() == "none":
            return None
        else:
            raise ValueError(f"未知的学习率调度器: {config.scheduler}")

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
        accumulated_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # 使用混合精度训练
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    loss = loss / self.config.training.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (
                    batch_idx + 1
                ) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()

                if (
                    batch_idx + 1
                ) % self.config.training.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 累计统计
            accumulated_loss += (
                loss.item() * self.config.training.gradient_accumulation_steps
            )
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            # 更新进度条
            if batch_idx % self.config.training.log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": accumulated_loss / total,
                        "acc": 100.0 * correct / total,
                    }
                )

        # 计算平均损失和准确率
        train_loss = accumulated_loss / len(self.train_loader.dataset)
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
            最佳验证准确率和最佳模型路径
        """
        self.start_time = time.time()
        best_model_path = str(self.save_dir / "best_model.pth")

        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%"
            )

            # 验证
            val_loss, val_acc = self.validate()
            logger.info(
                f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # 早停检查
            if self.early_stopping(val_loss, self.model, best_model_path):
                logger.info("Early stopping triggered")
                break

            # 更新最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), best_model_path)

        # 训练结束，加载最佳模型
        self.model.load_state_dict(torch.load(best_model_path))

        # 测试最佳模型
        test_loss, test_acc = self.test()
        logger.info(
            f"Best model (epoch {self.best_epoch}): "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

        return self.best_val_acc, best_model_path

    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制训练历史

        Args:
            save_path: 图像保存路径，如果为None则显示图像
        """
        plt.figure(figsize=(12, 4))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Train")
        plt.plot(self.val_losses, label="Validation")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label="Train")
        plt.plot(self.val_accs, label="Validation")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def predict_sample(
        self, data: torch.Tensor, classes: Optional[List[str]] = None
    ) -> Tuple[int, float]:
        """
        预测单个样本

        Args:
            data: 输入数据
            classes: 类别名称列表

        Returns:
            预测的类别索引和置信度
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            prob = torch.exp(output)
            pred = output.argmax(dim=1).item()
            confidence = prob[0][pred].item()

        return pred, confidence

    def get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要

        Returns:
            包含训练统计信息的字典
        """
        training_time = time.time() - self.start_time
        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "training_time": training_time,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }
