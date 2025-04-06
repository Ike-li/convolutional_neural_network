"""
可视化模块

提供各种可视化功能，包括训练过程、模型结构、特征图等
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from ..config.settings import (
    DEFAULT_FIGURE_SIZE,
    DEFAULT_DPI,
    DEFAULT_STYLE,
    DEFAULT_SAVE_FORMAT,
)


class Visualizer:
    """
    可视化器类

    提供各种可视化功能
    """

    def __init__(
        self,
        save_dir: str,
        figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
        dpi: int = DEFAULT_DPI,
        style: str = DEFAULT_STYLE,
        save_format: str = DEFAULT_SAVE_FORMAT,
    ) -> None:
        """
        初始化可视化器

        Args:
            save_dir: 图像保存目录
            figsize: 图像大小
            dpi: 图像DPI
            style: 绘图样式
            save_format: 保存格式
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.save_format = save_format
        os.makedirs(save_dir, exist_ok=True)

        # 设置绘图样式
        plt.style.use(style)

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_name: str = "training_history",
    ) -> None:
        """
        绘制训练历史

        Args:
            history: 训练历史字典
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for metric, values in history.items():
            plt.plot(values, label=metric)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: List[str],
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix",
    ) -> None:
        """
        绘制混淆矩阵

        Args:
            cm: 混淆矩阵
            classes: 类别名称列表
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_feature_maps(
        self,
        feature_maps: torch.Tensor,
        title: str = "Feature Maps",
        save_name: str = "feature_maps",
        max_channels: int = 16,
    ) -> None:
        """
        绘制特征图

        Args:
            feature_maps: 特征图张量 [B, C, H, W]
            title: 图像标题
            save_name: 保存文件名
            max_channels: 最大显示的通道数
        """
        # 转换为numpy数组
        feature_maps = feature_maps.detach().cpu().numpy()

        # 选择要显示的通道
        n_channels = min(feature_maps.shape[1], max_channels)
        feature_maps = feature_maps[0, :n_channels]  # 只显示第一个样本

        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(n_channels)))

        # 创建图像网格
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for i in range(n_channels):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(feature_maps[i], cmap="viridis")
            plt.axis("off")
            plt.title(f"Channel {i}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_model_structure(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        title: str = "Model Structure",
        save_name: str = "model_structure",
    ) -> None:
        """
        绘制模型结构

        Args:
            model: 模型
            input_shape: 输入形状
            title: 图像标题
            save_name: 保存文件名
        """
        try:
            from torchviz import make_dot

            # 创建示例输入
            x = torch.randn(1, *input_shape)
            y = model(x)

            # 创建计算图
            dot = make_dot(y, params=dict(model.named_parameters()))

            # 保存图像
            dot.render(os.path.join(self.save_dir, save_name), format=self.save_format)
        except ImportError:
            print("请安装torchviz以可视化模型结构")

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        title: str = "ROC Curve",
        save_name: str = "roc_curve",
    ) -> None:
        """
        绘制ROC曲线

        Args:
            fpr: 假阳性率
            tpr: 真阳性率
            roc_auc: ROC AUC分数
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        average_precision: float,
        title: str = "Precision-Recall Curve",
        save_name: str = "pr_curve",
    ) -> None:
        """
        绘制PR曲线

        Args:
            precision: 精确率
            recall: 召回率
            average_precision: 平均精确率
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(recall, precision, label=f"PR curve (AP = {average_precision:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Class Distribution",
        save_name: str = "class_distribution",
    ) -> None:
        """
        绘制类别分布

        Args:
            class_counts: 类别计数字典
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curve",
        save_name: str = "learning_curve",
    ) -> None:
        """
        绘制学习曲线

        Args:
            train_sizes: 训练集大小
            train_scores: 训练分数
            val_scores: 验证分数
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(train_sizes, train_scores, label="Training score")
        plt.plot(train_sizes, val_scores, label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_parameter_distribution(
        self,
        model: nn.Module,
        title: str = "Parameter Distribution",
        save_name: str = "parameter_distribution",
    ) -> None:
        """
        绘制参数分布

        Args:
            model: 模型
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for name, param in model.named_parameters():
            if param.requires_grad:
                plt.hist(
                    param.data.cpu().numpy().flatten(), bins=50, alpha=0.5, label=name
                )
        plt.xlabel("Parameter Value")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_gradient_flow(
        self,
        model: nn.Module,
        title: str = "Gradient Flow",
        save_name: str = "gradient_flow",
    ) -> None:
        """
        绘制梯度流

        Args:
            model: 模型
            title: 图像标题
            save_name: 保存文件名
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                plt.hist(
                    param.grad.data.cpu().numpy().flatten(),
                    bins=50,
                    alpha=0.5,
                    label=name,
                )
        plt.xlabel("Gradient Value")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_attention_maps(
        self,
        attention_maps: torch.Tensor,
        title: str = "Attention Maps",
        save_name: str = "attention_maps",
    ) -> None:
        """
        绘制注意力图

        Args:
            attention_maps: 注意力图张量
            title: 图像标题
            save_name: 保存文件名
        """
        # 转换为numpy数组
        attention_maps = attention_maps.detach().cpu().numpy()

        # 创建图像网格
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for i in range(attention_maps.shape[0]):
            plt.subplot(1, attention_maps.shape[0], i + 1)
            plt.imshow(attention_maps[i], cmap="viridis")
            plt.axis("off")
            plt.title(f"Head {i}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()

    def plot_sample_predictions(
        self,
        images: torch.Tensor,
        true_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        class_names: List[str],
        title: str = "Sample Predictions",
        save_name: str = "sample_predictions",
        num_samples: int = 16,
    ) -> None:
        """
        绘制样本预测结果

        Args:
            images: 图像张量
            true_labels: 真实标签
            pred_labels: 预测标签
            class_names: 类别名称列表
            title: 图像标题
            save_name: 保存文件名
            num_samples: 显示的样本数量
        """
        # 转换为numpy数组
        images = images.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()
        pred_labels = pred_labels.detach().cpu().numpy()

        # 选择要显示的样本
        num_samples = min(num_samples, len(images))
        images = images[:num_samples]
        true_labels = true_labels[:num_samples]
        pred_labels = pred_labels[:num_samples]

        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        # 创建图像网格
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for i in range(num_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            plt.title(
                f"True: {class_names[true_labels[i]]}\n"
                f"Pred: {class_names[pred_labels[i]]}",
                color="green" if true_labels[i] == pred_labels[i] else "red",
            )
            plt.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"{save_name}.{self.save_format}"),
            bbox_inches="tight",
        )
        plt.close()
