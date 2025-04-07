"""
可视化模块

本模块提供基本的可视化功能。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools
from sklearn.metrics import confusion_matrix


def plot_images(images, labels, predictions=None, classes=None, n=25):
    """
    显示图像样本

    Args:
        images: 图像张量
        labels: 实际标签
        predictions: 预测标签（可选）
        classes: 类别名称列表（可选）
        n: 显示的图像数量
    """
    # 确保n不超过图像数量
    n = min(n, len(images))

    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(n)))

    # 创建图
    plt.figure(figsize=(10, 10))

    for i in range(n):
        # 获取图像和标签
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.squeeze(img)  # 移除通道维度（对于灰度图像）
        label = labels[i].item()

        # 标准化到[0, 1]范围以正确显示
        img = (img - img.min()) / (img.max() - img.min())

        # 添加子图
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

        # 设置标题
        if classes:
            title = f"{classes[label]}"
            if predictions is not None:
                pred = predictions[i].item()
                title += f"\nPred: {classes[pred]}"
                # 如果预测错误，标记为红色
                if pred != label:
                    plt.title(title, color="red")
                else:
                    plt.title(title, color="green")
            else:
                plt.title(title)
        else:
            title = f"{label}"
            if predictions is not None:
                pred = predictions[i].item()
                title += f"\nPred: {pred}"
                # 如果预测错误，标记为红色
                if pred != label:
                    plt.title(title, color="red")
                else:
                    plt.title(title, color="green")
            else:
                plt.title(title)

    plt.tight_layout()
    plt.savefig("images.png")
    plt.close()


def plot_confusion_matrix(model, data_loader, device, classes=None):
    """
    计算并绘制混淆矩阵

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 计算设备
        classes: 类别名称列表（可选）
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("混淆矩阵")
    plt.colorbar()

    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    # 添加文本注释
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            str(int(cm[i, j])),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.savefig("confusion_matrix.png")
    plt.close()


def visualize_filters(model, layer_idx=0, n_filters=16):
    """
    可视化卷积层的过滤器

    Args:
        model: CNN模型
        layer_idx: 要可视化的卷积层索引
        n_filters: 要显示的过滤器数量
    """
    # 获取所有卷积层
    conv_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]

    if not conv_layers:
        print("模型中没有卷积层")
        return

    if layer_idx >= len(conv_layers):
        print(f"层索引超出范围，模型只有 {len(conv_layers)} 个卷积层")
        return

    # 获取指定层的权重
    weights = conv_layers[layer_idx].weight.data.cpu()
    n_filters = min(n_filters, weights.size(0))

    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    # 创建图
    plt.figure(figsize=(10, 10))

    for i in range(n_filters):
        # 获取第i个过滤器
        filter_weights = weights[i, 0]  # 假设输入是单通道图像

        # 标准化到[0, 1]范围以正确显示
        filter_weights = (filter_weights - filter_weights.min()) / (
            filter_weights.max() - filter_weights.min()
        )

        # 添加子图
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(filter_weights, cmap="viridis")
        plt.axis("off")
        plt.title(f"Filter {i + 1}")

    plt.suptitle(f"卷积层 {layer_idx + 1} 的过滤器")
    plt.tight_layout()
    plt.savefig("filters.png")  # 保存图像以满足测试要求
    plt.show()


def visualize_feature_maps(model, data, device, layer_idx=0, n_maps=16):
    """
    可视化特征图

    Args:
        model: CNN模型
        data: 输入图像批次
        device: 计算设备
        layer_idx: 要可视化的卷积层索引
        n_maps: 要显示的特征图数量
    """
    model.eval()

    # 获取所有卷积层
    conv_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ]

    if not conv_layers:
        print("模型中没有卷积层")
        return

    if layer_idx >= len(conv_layers):
        print(f"层索引超出范围，模型只有 {len(conv_layers)} 个卷积层")
        return

    # 选择一张图像
    img = data[0:1].to(device)

    # 钩子函数，用于保存特征图
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 注册钩子
    target_layer = conv_layers[layer_idx]
    handle = target_layer.register_forward_hook(get_activation("target_layer"))

    # 前向传播
    with torch.no_grad():
        model(img)

    # 移除钩子
    handle.remove()

    # 获取特征图
    feature_maps = activation["target_layer"]
    feature_maps = feature_maps.squeeze(0)

    # 可视化特征图
    n_maps = min(n_maps, feature_maps.size(0))
    grid_size = int(np.ceil(np.sqrt(n_maps)))

    plt.figure(figsize=(10, 10))

    for i in range(n_maps):
        feature_map = feature_maps[i].cpu().numpy()

        # 标准化到[0, 1]范围以正确显示
        feature_map = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min() + 1e-8
        )

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(feature_map, cmap="viridis")
        plt.axis("off")
        plt.title(f"Map {i + 1}")

    plt.suptitle(f"卷积层 {layer_idx + 1} 的特征图")
    plt.tight_layout()
    plt.savefig("feature_maps.png")  # 保存图像以满足测试要求
    plt.show()
