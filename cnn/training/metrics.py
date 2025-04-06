"""
指标模块

提供各种评估指标的计算功能
"""

import numpy as np
import torch
from typing import Dict, List

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


class Metric:
    """
    指标基类

    所有指标的基类，提供基本的指标接口
    """

    def __init__(self, name: str) -> None:
        """
        初始化指标

        Args:
            name: 指标名称
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        pass

    def update(self, *args, **kwargs) -> None:
        """
        更新指标状态

        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        raise NotImplementedError

    def compute(self) -> float:
        """
        计算指标值

        Returns:
            指标值
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """返回指标的字符串表示"""
        return f"{self.name}: {self.compute():.4f}"


class Accuracy(Metric):
    """
    准确率指标

    计算分类准确率
    """

    def __init__(self) -> None:
        """初始化准确率指标"""
        super().__init__("accuracy")
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值
            targets: 目标值
        """
        preds = preds.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        """
        计算准确率

        Returns:
            准确率
        """
        return self.correct / self.total if self.total > 0 else 0.0


class Precision(Metric):
    """
    精确率指标

    计算分类精确率
    """

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        """
        初始化精确率指标

        Args:
            num_classes: 类别数量
            average: 平均方式 ("macro", "micro", "weighted")
        """
        super().__init__("precision")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值
            targets: 目标值
        """
        preds = preds.argmax(dim=1)
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        """
        计算精确率

        Returns:
            精确率
        """
        return precision_score(
            self.targets,
            self.preds,
            average=self.average,
            labels=range(self.num_classes),
        )


class Recall(Metric):
    """
    召回率指标

    计算分类召回率
    """

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        """
        初始化召回率指标

        Args:
            num_classes: 类别数量
            average: 平均方式 ("macro", "micro", "weighted")
        """
        super().__init__("recall")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值
            targets: 目标值
        """
        preds = preds.argmax(dim=1)
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        """
        计算召回率

        Returns:
            召回率
        """
        return recall_score(
            self.targets,
            self.preds,
            average=self.average,
            labels=range(self.num_classes),
        )


class F1Score(Metric):
    """
    F1分数指标

    计算分类F1分数
    """

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        """
        初始化F1分数指标

        Args:
            num_classes: 类别数量
            average: 平均方式 ("macro", "micro", "weighted")
        """
        super().__init__("f1")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值
            targets: 目标值
        """
        preds = preds.argmax(dim=1)
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        """
        计算F1分数

        Returns:
            F1分数
        """
        return f1_score(
            self.targets,
            self.preds,
            average=self.average,
            labels=range(self.num_classes),
        )


class ConfusionMatrix(Metric):
    """
    混淆矩阵指标

    计算分类混淆矩阵
    """

    def __init__(self, num_classes: int) -> None:
        """
        初始化混淆矩阵指标

        Args:
            num_classes: 类别数量
        """
        super().__init__("confusion_matrix")
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值
            targets: 目标值
        """
        preds = preds.argmax(dim=1)
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> np.ndarray:
        """
        计算混淆矩阵

        Returns:
            混淆矩阵
        """
        return confusion_matrix(
            self.targets,
            self.preds,
            labels=range(self.num_classes),
        )


class ROCAUC(Metric):
    """
    ROC AUC指标

    计算分类ROC AUC分数
    """

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        """
        初始化ROC AUC指标

        Args:
            num_classes: 类别数量
            average: 平均方式 ("macro", "micro", "weighted", "ovr")
        """
        super().__init__("roc_auc")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值（概率）
            targets: 目标值
        """
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        """
        计算ROC AUC分数

        Returns:
            ROC AUC分数
        """
        return roc_auc_score(
            self.targets,
            self.preds,
            average=self.average,
            multi_class="ovr",
        )


class AveragePrecision(Metric):
    """
    平均精确率指标

    计算分类平均精确率
    """

    def __init__(self, num_classes: int, average: str = "macro") -> None:
        """
        初始化平均精确率指标

        Args:
            num_classes: 类别数量
            average: 平均方式 ("macro", "micro", "weighted")
        """
        super().__init__("average_precision")
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self) -> None:
        """重置指标状态"""
        self.preds = []
        self.targets = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新指标状态

        Args:
            preds: 预测值（概率）
            targets: 目标值
        """
        self.preds.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> float:
        """
        计算平均精确率

        Returns:
            平均精确率
        """
        return average_precision_score(
            self.targets,
            self.preds,
            average=self.average,
        )


class MetricCollection:
    """
    指标集合

    管理多个指标的计算
    """

    def __init__(self, metrics: List[Metric]) -> None:
        """
        初始化指标集合

        Args:
            metrics: 指标列表
        """
        self.metrics = metrics

    def reset(self) -> None:
        """重置所有指标"""
        for metric in self.metrics:
            metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新所有指标

        Args:
            preds: 预测值
            targets: 目标值
        """
        for metric in self.metrics:
            metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """
        计算所有指标

        Returns:
            指标名称到指标值的映射
        """
        return {metric.name: metric.compute() for metric in self.metrics}

    def __str__(self) -> str:
        """返回所有指标的字符串表示"""
        return "\n".join(str(metric) for metric in self.metrics)
