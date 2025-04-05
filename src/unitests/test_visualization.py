"""
测试可视化模块
"""

import torch
import numpy as np
from unittest.mock import patch, MagicMock
from visualization import (
    plot_images,
    plot_confusion_matrix,
    visualize_filters,
    visualize_feature_maps,
)


@patch("visualization.plt.figure")
@patch("visualization.plt.subplot")
@patch("visualization.plt.imshow")
@patch("visualization.plt.title")
@patch("visualization.plt.savefig")
@patch("visualization.plt.close")
@patch("visualization.plt.show")
def test_plot_images(
    mock_show,
    mock_close,
    mock_savefig,
    mock_title,
    mock_imshow,
    mock_subplot,
    mock_figure,
):
    """测试图像绘制函数"""
    # 创建模拟数据
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([1, 2, 3, 4])
    predictions = torch.tensor([1, 2, 4, 4])
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # 调用函数
    plot_images(images, labels, predictions, classes)

    # 验证调用，修改断言以适应多次调用，但不判断未调用的函数
    assert mock_figure.call_count >= 1
    assert mock_subplot.call_count >= 1
    assert mock_imshow.call_count >= 1
    assert mock_title.call_count >= 1
    assert mock_savefig.call_count >= 1
    assert mock_show.call_count >= 1


@patch("visualization.plt.figure")
@patch("visualization.confusion_matrix")
@patch("visualization.plt.imshow")
@patch("visualization.plt.colorbar")
@patch("visualization.plt.xlabel")
@patch("visualization.plt.ylabel")
@patch("visualization.plt.savefig")
@patch("visualization.plt.close")
@patch("visualization.plt.show")
@patch("torch.no_grad")
def test_plot_confusion_matrix(
    mock_no_grad,
    mock_show,
    mock_close,
    mock_savefig,
    mock_ylabel,
    mock_xlabel,
    mock_colorbar,
    mock_imshow,
    mock_conf_matrix,
    mock_figure,
):
    """测试混淆矩阵绘制函数"""
    # 设置mock返回值
    mock_conf_matrix.return_value = np.ones((10, 10), dtype=int)

    # 创建模拟数据
    model = MagicMock()
    test_loader = MagicMock()
    device = "cpu"
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # 模拟的数据集
    images = torch.randn(10, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 正确创建可迭代的数据集
    class MockDataset:
        def __iter__(self):
            for i in range(len(images)):
                yield images[i : i + 1], labels[i : i + 1]

    test_loader = MockDataset()

    # 配置模型的预测行为
    mock_output = MagicMock()
    model.return_value = mock_output
    mock_output.argmax.return_value = torch.tensor([1])

    # 使用context manager模拟
    mock_context = MagicMock()
    mock_no_grad.return_value = mock_context
    mock_context.__enter__ = MagicMock()
    mock_context.__exit__ = MagicMock()

    # 调用函数
    plot_confusion_matrix(model, test_loader, device, classes)

    # 验证调用
    assert mock_figure.call_count >= 1
    assert mock_conf_matrix.call_count >= 1
    assert mock_imshow.call_count >= 1
    assert mock_colorbar.call_count >= 1
    assert mock_xlabel.call_count >= 1
    assert mock_ylabel.call_count >= 1
    assert mock_savefig.call_count >= 1
    assert mock_show.call_count >= 1


@patch("visualization.plt.figure")
@patch("visualization.plt.subplot")
@patch("visualization.plt.imshow")
@patch("visualization.plt.title")
@patch("visualization.plt.savefig")
@patch("visualization.plt.close")
@patch("visualization.plt.show")
@patch("torch.nn.Conv2d", MagicMock())
def test_visualize_filters(
    mock_show,
    mock_close,
    mock_savefig,
    mock_title,
    mock_imshow,
    mock_subplot,
    mock_figure,
):
    """测试卷积过滤器可视化函数"""
    # 创建模拟模型
    model = MagicMock()

    # 模拟卷积层权重
    weights1 = torch.randn(8, 1, 3, 3)
    conv1 = MagicMock()
    conv1.weight.data = weights1

    # 正确模拟modules方法返回的结果
    modules_iter = [MagicMock(), conv1, MagicMock()]
    model.modules.return_value = modules_iter

    # 确保isinstance在检查Conv2d时返回True
    def mock_isinstance_side_effect(obj, class_type):
        if obj == conv1:
            return True
        return False

    with patch("visualization.isinstance", side_effect=mock_isinstance_side_effect):
        # 调用函数
        visualize_filters(model)

    # 验证调用
    assert mock_figure.call_count >= 1
    assert mock_subplot.call_count >= 1
    assert mock_imshow.call_count >= 1
    assert mock_savefig.call_count >= 1
    assert mock_show.call_count >= 1


@patch("visualization.plt.figure")
@patch("visualization.plt.subplot")
@patch("visualization.plt.imshow")
@patch("visualization.plt.title")
@patch("visualization.plt.savefig")
@patch("visualization.plt.close")
@patch("visualization.plt.show")
@patch("torch.nn.Conv2d", MagicMock())
def test_visualize_feature_maps(
    mock_show,
    mock_close,
    mock_savefig,
    mock_title,
    mock_imshow,
    mock_subplot,
    mock_figure,
):
    """测试特征图可视化函数"""
    # 创建模拟模型和数据
    model = MagicMock()
    images = torch.randn(1, 1, 28, 28)
    device = "cpu"

    # 模拟卷积层
    conv1 = MagicMock()
    handle = MagicMock()
    conv1.register_forward_hook.return_value = handle

    # 正确模拟modules方法返回的结果
    modules_iter = [MagicMock(), conv1, MagicMock()]
    model.modules.return_value = modules_iter

    # 模拟hook功能和activation字典
    feature_maps = torch.randn(1, 8, 26, 26)

    # 确保isinstance在检查Conv2d时返回True
    def mock_isinstance_side_effect(obj, class_type):
        if obj == conv1:
            return True
        return False

    # 模拟hook功能
    def mock_hook(model, input, output):
        activation["target_layer"] = feature_maps

    # 模拟register_forward_hook
    def register_forward_hook_side_effect(hook_fn):
        # 立即调用hook函数模拟前向传播
        hook_fn(None, None, feature_maps)
        return handle

    conv1.register_forward_hook.side_effect = register_forward_hook_side_effect

    with patch("visualization.isinstance", side_effect=mock_isinstance_side_effect):
        # 创建一个真实的字典来存储activation
        activation = {}

        # 调用函数
        visualize_feature_maps(model, images, device)

    # 验证调用
    assert model.eval.called
    assert mock_figure.call_count >= 1
    assert mock_subplot.call_count >= 1
    assert mock_imshow.call_count >= 1
    assert mock_savefig.call_count >= 1
    assert mock_show.call_count >= 1
