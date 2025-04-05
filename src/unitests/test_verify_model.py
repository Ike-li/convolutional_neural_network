"""
测试模型验证模块
"""

import pytest
from unittest.mock import patch, MagicMock
from verify_model import verify_model


@patch('verify_model.traceback.print_exc')
@patch('verify_model.check_feature_map_sizes')
@patch('verify_model.detailed_model_analysis')
@patch('verify_model.SimpleCNN')
@patch('verify_model.DeepCNN')
@patch('verify_model.get_mnist_loaders')
@patch('verify_model.get_fashion_mnist_loaders')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.device')
def test_verify_model_simple_mnist(mock_device, mock_cuda, 
                                  mock_fashion_loaders, mock_mnist_loaders,
                                  mock_deep, mock_simple, 
                                  mock_detailed, mock_check_feature, mock_traceback):
    """测试verify_model函数使用simple模型和mnist数据集"""
    # 设置mocks
    mock_device.return_value = 'cpu'
    
    # 设置模型mock
    mock_model = MagicMock()
    mock_simple.return_value = mock_model
    mock_model.to.return_value = mock_model
    
    # 设置模型分析mock
    mock_detailed.return_value = {}
    
    # 设置数据加载器mock
    mock_images = MagicMock()
    mock_labels = MagicMock()
    mock_batch = [mock_images, mock_labels]
    mock_data_iter = MagicMock()
    mock_data_iter.__iter__.return_value = iter([mock_batch])
    
    mock_mnist_loaders.return_value = (mock_data_iter, None, None)
    
    # 设置tensor操作mocks
    mock_images.to.return_value = mock_images
    mock_labels.to.return_value = mock_labels
    mock_output = MagicMock()
    mock_model.return_value = mock_output
    
    # 执行函数
    verify_model(model_name="simple", dataset_name="mnist")
    
    # 验证调用
    mock_simple.assert_called_once()
    mock_deep.assert_not_called()
    mock_mnist_loaders.assert_called_once()
    mock_fashion_loaders.assert_not_called()
    mock_detailed.assert_called_once()
    mock_model.eval.assert_called_once()


@patch('verify_model.traceback.print_exc')
@patch('verify_model.check_feature_map_sizes')
@patch('verify_model.detailed_model_analysis')
@patch('verify_model.SimpleCNN')
@patch('verify_model.DeepCNN')
@patch('verify_model.get_mnist_loaders')
@patch('verify_model.get_fashion_mnist_loaders')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.device')
def test_verify_model_deep_fashion(mock_device, mock_cuda, 
                                  mock_fashion_loaders, mock_mnist_loaders,
                                  mock_deep, mock_simple, 
                                  mock_detailed, mock_check_feature, mock_traceback):
    """测试verify_model函数使用deep模型和fashion_mnist数据集"""
    # 设置mocks
    mock_device.return_value = 'cpu'
    
    # 设置模型mock
    mock_model = MagicMock()
    mock_deep.return_value = mock_model
    mock_model.to.return_value = mock_model
    
    # 设置模型分析mock
    mock_detailed.return_value = {}
    
    # 设置数据加载器mock
    mock_images = MagicMock()
    mock_labels = MagicMock()
    mock_batch = [mock_images, mock_labels]
    mock_data_iter = MagicMock()
    mock_data_iter.__iter__.return_value = iter([mock_batch])
    
    mock_fashion_loaders.return_value = (mock_data_iter, None, None)
    
    # 设置tensor操作mocks
    mock_images.to.return_value = mock_images
    mock_labels.to.return_value = mock_labels
    mock_output = MagicMock()
    mock_model.return_value = mock_output
    
    # 执行函数
    verify_model(model_name="deep", dataset_name="fashion_mnist")
    
    # 验证调用
    mock_simple.assert_not_called()
    mock_deep.assert_called_once()
    mock_mnist_loaders.assert_not_called()
    mock_fashion_loaders.assert_called_once()
    mock_detailed.assert_called_once()
    mock_model.eval.assert_called_once() 