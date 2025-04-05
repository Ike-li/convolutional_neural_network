"""
测试工具模块
"""

from utils import check_feature_map_sizes, detailed_model_analysis
from models import SimpleCNN


def test_check_feature_map_sizes(capsys):
    """测试特征图尺寸检查函数"""
    model = SimpleCNN()
    check_feature_map_sizes(model)

    # 捕获并验证函数输出
    captured = capsys.readouterr()
    assert "输入尺寸" in captured.out
    assert "各层的输出尺寸" in captured.out
    assert "最终输出尺寸" in captured.out


def test_detailed_model_analysis(capsys):
    """测试模型详细分析函数"""
    model = SimpleCNN()
    result = detailed_model_analysis(model)

    # 验证函数返回值
    assert isinstance(result, dict)
    assert len(result) > 0

    # 捕获并验证函数输出
    captured = capsys.readouterr()
    assert "模型详细分析" in captured.out
    assert "总参数量" in captured.out
    assert "输入形状" in captured.out
    assert "最终输出形状" in captured.out
