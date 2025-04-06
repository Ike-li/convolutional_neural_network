"""
日志模块

提供日志记录功能，包括控制台输出和文件记录
"""

import os
import logging
from typing import Optional, Dict, Any

from ..config.settings import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FILE,
)


class Logger:
    """
    日志记录器类

    提供日志记录功能，包括控制台输出和文件记录
    """

    def __init__(
        self,
        name: str,
        log_dir: str,
        log_file: str = DEFAULT_LOG_FILE,
        log_format: str = DEFAULT_LOG_FORMAT,
        log_level: int = DEFAULT_LOG_LEVEL,
    ) -> None:
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志文件目录
            log_file: 日志文件名
            log_format: 日志格式
            log_level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_format = log_format
        self.log_level = log_level

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # 创建格式化器
        formatter = logging.Formatter(log_format)

        # 创建文件处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file),
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        """
        记录信息级别的日志

        Args:
            message: 日志消息
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        记录警告级别的日志

        Args:
            message: 日志消息
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        记录错误级别的日志

        Args:
            message: 日志消息
        """
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """
        记录调试级别的日志

        Args:
            message: 日志消息
        """
        self.logger.debug(message)

    def critical(self, message: str) -> None:
        """
        记录严重错误级别的日志

        Args:
            message: 日志消息
        """
        self.logger.critical(message)

    def log_training_start(
        self,
        model_name: str,
        dataset_name: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
    ) -> None:
        """
        记录训练开始信息

        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 训练设备
        """
        self.info("=" * 50)
        self.info("训练开始")
        self.info("=" * 50)
        self.info(f"模型名称: {model_name}")
        self.info(f"数据集名称: {dataset_name}")
        self.info(f"训练轮数: {num_epochs}")
        self.info(f"批次大小: {batch_size}")
        self.info(f"学习率: {learning_rate}")
        self.info(f"训练设备: {device}")
        self.info("=" * 50)

    def log_training_end(
        self,
        total_time: float,
        best_epoch: int,
        best_val_loss: float,
        best_val_acc: float,
    ) -> None:
        """
        记录训练结束信息

        Args:
            total_time: 总训练时间
            best_epoch: 最佳轮数
            best_val_loss: 最佳验证损失
            best_val_acc: 最佳验证准确率
        """
        self.info("=" * 50)
        self.info("训练结束")
        self.info("=" * 50)
        self.info(f"总训练时间: {total_time:.2f}秒")
        self.info(f"最佳轮数: {best_epoch}")
        self.info(f"最佳验证损失: {best_val_loss:.4f}")
        self.info(f"最佳验证准确率: {best_val_acc:.4f}")
        self.info("=" * 50)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        time_taken: float,
    ) -> None:
        """
        记录每个轮次的训练信息

        Args:
            epoch: 当前轮数
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            time_taken: 耗时
        """
        self.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {time_taken:.2f}s"
        )

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        acc: float,
        time_taken: float,
    ) -> None:
        """
        记录每个批次的训练信息

        Args:
            epoch: 当前轮数
            batch: 当前批次
            total_batches: 总批次数
            loss: 损失值
            acc: 准确率
            time_taken: 耗时
        """
        self.debug(
            f"Epoch {epoch:3d} | "
            f"Batch {batch:3d}/{total_batches:3d} | "
            f"Loss: {loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Time: {time_taken:.2f}s"
        )

    def log_model_summary(self, model_summary: Dict[str, Any]) -> None:
        """
        记录模型摘要信息

        Args:
            model_summary: 模型摘要字典
        """
        self.info("=" * 50)
        self.info("模型摘要")
        self.info("=" * 50)
        for key, value in model_summary.items():
            self.info(f"{key}: {value}")
        self.info("=" * 50)

    def log_test_results(
        self,
        test_loss: float,
        test_acc: float,
        class_acc: Dict[str, float],
        confusion_matrix: Any,
    ) -> None:
        """
        记录测试结果

        Args:
            test_loss: 测试损失
            test_acc: 测试准确率
            class_acc: 每个类别的准确率
            confusion_matrix: 混淆矩阵
        """
        self.info("=" * 50)
        self.info("测试结果")
        self.info("=" * 50)
        self.info(f"测试损失: {test_loss:.4f}")
        self.info(f"测试准确率: {test_acc:.4f}")
        self.info("各类别准确率:")
        for class_name, acc in class_acc.items():
            self.info(f"  {class_name}: {acc:.4f}")
        self.info("混淆矩阵:")
        self.info(str(confusion_matrix))
        self.info("=" * 50)

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        记录错误信息

        Args:
            error: 异常对象
            context: 错误上下文
        """
        if context:
            self.error(f"错误上下文: {context}")
        self.error(f"错误类型: {type(error).__name__}")
        self.error(f"错误信息: {str(error)}")
        self.error("错误堆栈:", exc_info=True)

    def log_warning(self, message: str, context: Optional[str] = None) -> None:
        """
        记录警告信息

        Args:
            message: 警告消息
            context: 警告上下文
        """
        if context:
            self.warning(f"警告上下文: {context}")
        self.warning(message)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """
        记录超参数信息

        Args:
            hyperparameters: 超参数字典
        """
        self.info("=" * 50)
        self.info("超参数")
        self.info("=" * 50)
        for key, value in hyperparameters.items():
            self.info(f"{key}: {value}")
        self.info("=" * 50)

    def log_save_checkpoint(
        self,
        epoch: int,
        model_path: str,
        optimizer_path: str,
        scheduler_path: Optional[str] = None,
    ) -> None:
        """
        记录保存检查点信息

        Args:
            epoch: 当前轮数
            model_path: 模型保存路径
            optimizer_path: 优化器保存路径
            scheduler_path: 调度器保存路径
        """
        self.info(f"保存检查点 (Epoch {epoch})")
        self.info(f"模型保存路径: {model_path}")
        self.info(f"优化器保存路径: {optimizer_path}")
        if scheduler_path:
            self.info(f"调度器保存路径: {scheduler_path}")

    def log_load_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        best_val_loss: float,
        best_val_acc: float,
    ) -> None:
        """
        记录加载检查点信息

        Args:
            checkpoint_path: 检查点路径
            epoch: 轮数
            best_val_loss: 最佳验证损失
            best_val_acc: 最佳验证准确率
        """
        self.info("=" * 50)
        self.info("加载检查点")
        self.info("=" * 50)
        self.info(f"检查点路径: {checkpoint_path}")
        self.info(f"轮数: {epoch}")
        self.info(f"最佳验证损失: {best_val_loss:.4f}")
        self.info(f"最佳验证准确率: {best_val_acc:.4f}")
        self.info("=" * 50)

    def log_early_stopping(
        self,
        epoch: int,
        patience: int,
        best_val_loss: float,
        best_epoch: int,
    ) -> None:
        """
        记录早停信息

        Args:
            epoch: 当前轮数
            patience: 早停耐心值
            best_val_loss: 最佳验证损失
            best_epoch: 最佳轮数
        """
        self.info("=" * 50)
        self.info("早停触发")
        self.info("=" * 50)
        self.info(f"当前轮数: {epoch}")
        self.info(f"早停耐心值: {patience}")
        self.info(f"最佳验证损失: {best_val_loss:.4f}")
        self.info(f"最佳轮数: {best_epoch}")
        self.info("=" * 50)

    def log_learning_rate(self, epoch: int, learning_rate: float) -> None:
        """
        记录学习率信息

        Args:
            epoch: 当前轮数
            learning_rate: 当前学习率
        """
        self.info(f"Epoch {epoch:3d} | Learning Rate: {learning_rate:.6f}")

    def log_memory_usage(self) -> None:
        """
        记录内存使用情况
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            self.info(
                f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB | "
                f"虚拟内存: {memory_info.vms / 1024 / 1024:.2f} MB"
            )
        except ImportError:
            self.warning("请安装psutil以监控内存使用情况")

    def log_gpu_usage(self) -> None:
        """
        记录GPU使用情况
        """
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    memory_cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                    self.info(
                        f"GPU {i} | "
                        f"已分配内存: {memory_allocated:.2f} MB | "
                        f"缓存内存: {memory_cached:.2f} MB"
                    )
        except Exception as e:
            self.warning(f"无法获取GPU使用情况: {str(e)}")

    def log_system_info(self) -> None:
        """
        记录系统信息
        """
        import platform
        import torch

        self.info("=" * 50)
        self.info("系统信息")
        self.info("=" * 50)
        self.info(f"操作系统: {platform.platform()}")
        self.info(f"Python版本: {platform.python_version()}")
        self.info(f"PyTorch版本: {torch.__version__}")
        self.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.info(f"CUDA版本: {torch.version.cuda}")
            self.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        self.info("=" * 50)
