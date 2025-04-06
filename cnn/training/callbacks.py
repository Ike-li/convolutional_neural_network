"""
回调模块

提供训练过程中的回调功能，用于监控和记录训练状态
"""

import os
import logging
from typing import Optional, List

import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


class BaseCallback:
    """
    回调基类

    所有回调类的基类，提供基本的回调接口
    """

    def __init__(self) -> None:
        """初始化回调"""
        pass

    def on_train_begin(self, **kwargs) -> None:
        """训练开始时调用"""
        pass

    def on_train_end(self, **kwargs) -> None:
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """每个epoch开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """每个epoch结束时调用"""
        pass

    def on_batch_begin(self, batch: int, **kwargs) -> None:
        """每个batch开始时调用"""
        pass

    def on_batch_end(self, batch: int, **kwargs) -> None:
        """每个batch结束时调用"""
        pass


class ModelCheckpoint(BaseCallback):
    """
    模型检查点回调

    在训练过程中保存模型
    """

    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        save_freq: int = 1,
        monitor: str = "val_acc",
        mode: str = "max",
    ) -> None:
        """
        初始化模型检查点回调

        Args:
            save_dir: 模型保存目录
            save_best_only: 是否只保存最佳模型
            save_freq: 保存频率（每隔多少个epoch）
            monitor: 监控指标
            mode: 监控模式 ("min" 或 "max")
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """
        每个epoch结束时调用

        Args:
            epoch: 当前epoch数
            **kwargs: 其他参数，必须包含monitor指定的指标
        """
        if self.monitor not in kwargs:
            logger.warning(f"监控指标 {self.monitor} 不存在")
            return

        current_value = kwargs[self.monitor]
        should_save = False

        if self.mode == "min":
            if current_value < self.best_value:
                self.best_value = current_value
                should_save = True
        else:  # mode == "max"
            if current_value > self.best_value:
                self.best_value = current_value
                should_save = True

        if should_save or (not self.save_best_only and epoch % self.save_freq == 0):
            model = kwargs.get("model")
            if model is not None:
                save_path = os.path.join(
                    self.save_dir,
                    f"model_epoch_{epoch}_{self.monitor}_{current_value:.4f}.pth",
                )
                model.save(save_path)
                logger.info(f"保存模型到 {save_path}")


class EarlyStopping(BaseCallback):
    """
    早停回调

    当模型性能不再提升时停止训练
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        """
        初始化早停回调

        Args:
            monitor: 监控指标
            min_delta: 最小改善值
            patience: 容忍多少个epoch没有改善
            mode: 监控模式 ("min" 或 "max")
            restore_best_weights: 是否恢复最佳权重
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, **kwargs) -> None:
        """训练开始时调用"""
        self.wait = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """
        每个epoch结束时调用

        Args:
            epoch: 当前epoch数
            **kwargs: 其他参数，必须包含monitor指定的指标
        """
        if self.monitor not in kwargs:
            logger.warning(f"监控指标 {self.monitor} 不存在")
            return

        current_value = kwargs[self.monitor]
        model = kwargs.get("model")

        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.wait = 0
                if self.restore_best_weights and model is not None:
                    self.best_weights = model.state_dict()
            else:
                self.wait += 1
        else:  # mode == "max"
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.wait = 0
                if self.restore_best_weights and model is not None:
                    self.best_weights = model.state_dict()
            else:
                self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if model is not None and self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            logger.info(f"早停触发！{self.patience} 个epoch未改善")
            return True  # 返回True表示应该停止训练

        return False


class TensorBoardCallback(BaseCallback):
    """
    TensorBoard回调

    将训练指标记录到TensorBoard
    """

    def __init__(self, log_dir: str) -> None:
        """
        初始化TensorBoard回调

        Args:
            log_dir: 日志目录
        """
        super().__init__()
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logger.warning("TensorBoard未安装，禁用TensorBoard回调")
            self.enabled = False

    def on_train_begin(self, **kwargs) -> None:
        """训练开始时调用"""
        if not self.enabled:
            return
        model = kwargs.get("model")
        if model is not None:
            # 记录模型图
            dummy_input = torch.randn(1, 1, 28, 28)
            self.writer.add_graph(model, dummy_input)

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """
        每个epoch结束时调用

        Args:
            epoch: 当前epoch数
            **kwargs: 其他参数，包含各种训练指标
        """
        if not self.enabled:
            return

        # 记录标量指标
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # 记录学习率
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                self.writer.add_scalar(
                    f"learning_rate/group_{i}", param_group["lr"], epoch
                )

    def on_train_end(self, **kwargs) -> None:
        """训练结束时调用"""
        if self.enabled:
            self.writer.close()


class CSVLogger(BaseCallback):
    """
    CSV日志回调

    将训练指标记录到CSV文件
    """

    def __init__(self, filename: str) -> None:
        """
        初始化CSV日志回调

        Args:
            filename: CSV文件名
        """
        super().__init__()
        self.filename = filename
        self.header = None
        self.file = None
        self.metrics = []

    def on_train_begin(self, **kwargs) -> None:
        """训练开始时调用"""
        self.metrics = []
        self.file = open(self.filename, "w")

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """
        每个epoch结束时调用

        Args:
            epoch: 当前epoch数
            **kwargs: 其他参数，包含各种训练指标
        """
        metrics = {"epoch": epoch}
        metrics.update({k: v for k, v in kwargs.items() if isinstance(v, (int, float))})
        self.metrics.append(metrics)

        # 写入CSV
        if self.header is None:
            self.header = list(metrics.keys())
            self.file.write(",".join(self.header) + "\n")
        self.file.write(",".join(str(metrics[k]) for k in self.header) + "\n")
        self.file.flush()

    def on_train_end(self, **kwargs) -> None:
        """训练结束时调用"""
        if self.file is not None:
            self.file.close()


class PlotCallback(BaseCallback):
    """
    绘图回调

    绘制训练过程中的各种指标
    """

    def __init__(
        self,
        save_dir: str,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (10, 6),
        dpi: int = 100,
    ) -> None:
        """
        初始化绘图回调

        Args:
            save_dir: 图像保存目录
            metrics: 要绘制的指标列表
            figsize: 图像大小
            dpi: 图像DPI
        """
        super().__init__()
        self.save_dir = save_dir
        self.metrics = metrics or ["loss", "acc"]
        self.figsize = figsize
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        self.history = {metric: [] for metric in self.metrics}

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """
        每个epoch结束时调用

        Args:
            epoch: 当前epoch数
            **kwargs: 其他参数，包含各种训练指标
        """
        # 记录指标
        for metric in self.metrics:
            if metric in kwargs:
                self.history[metric].append(kwargs[metric])

        # 绘制图像
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for metric in self.metrics:
            if self.history[metric]:
                plt.plot(self.history[metric], label=metric)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f"metrics_epoch_{epoch}.png"))
        plt.close()


class CallbackList:
    """
    回调列表

    管理多个回调的执行
    """

    def __init__(self, callbacks: List[BaseCallback]) -> None:
        """
        初始化回调列表

        Args:
            callbacks: 回调列表
        """
        self.callbacks = callbacks

    def on_train_begin(self, **kwargs) -> None:
        """训练开始时调用所有回调"""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs) -> None:
        """训练结束时调用所有回调"""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """每个epoch开始时调用所有回调"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """每个epoch结束时调用所有回调"""
        for callback in self.callbacks:
            if callback.on_epoch_end(epoch, **kwargs):
                return True  # 如果任何回调返回True，表示应该停止训练
        return False

    def on_batch_begin(self, batch: int, **kwargs) -> None:
        """每个batch开始时调用所有回调"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)

    def on_batch_end(self, batch: int, **kwargs) -> None:
        """每个batch结束时调用所有回调"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, **kwargs)
