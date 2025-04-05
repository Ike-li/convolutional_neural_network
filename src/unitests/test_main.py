"""
测试主模块
"""

import pytest
import argparse
import os.path
from main import parse_args, train_and_evaluate
from unittest.mock import patch, MagicMock


def test_parse_args():
    """测试命令行参数解析函数"""
    with patch('argparse.ArgumentParser.parse_args', 
               return_value=argparse.Namespace(
                   dataset='mnist',
                   model='simple',
                   epochs=5,
                   batch_size=64,
                   test_batch_size=100,
                   lr=0.01,
                   momentum=0.5,
                   optimizer='sgd',
                   weight_decay=0.0,
                   log_interval=10,
                   save_dir='models',
                   data_dir='data',
                   augment=False,
                   visualize=False,
                   val_ratio=0.1,
                   no_cuda=True,
                   seed=42
               )):
        args = parse_args()
        assert args.dataset == 'mnist'
        assert args.model == 'simple'
        assert args.epochs == 5
        assert args.batch_size == 64
        assert args.test_batch_size == 100
        assert args.lr == 0.01
        assert args.momentum == 0.5
        assert args.optimizer == 'sgd'
        assert args.weight_decay == 0.0
        assert args.log_interval == 10
        assert args.save_dir == 'models'
        assert args.data_dir == 'data'
        assert args.augment is False
        assert args.visualize is False
        assert args.val_ratio == 0.1
        assert args.no_cuda is True
        assert args.seed == 42
        

@patch('torch.device')
@patch('torch.cuda.is_available', return_value=False)
@patch('main.get_data_loaders')
@patch('main.get_dataset_classes')
@patch('main.get_model')
@patch('main.Trainer')
@patch('os.makedirs')
@patch('os.path.exists', return_value=True)
@patch('torch.load')
def test_train_and_evaluate(mock_torch_load, mock_path_exists, mock_makedirs, 
                            mock_trainer, mock_get_model, mock_get_classes, 
                            mock_get_loaders, mock_cuda, mock_device):
    """测试训练和评估函数"""
    # 设置mock对象
    args = argparse.Namespace(
        dataset='mnist',
        model='simple',
        epochs=1,
        batch_size=64,
        test_batch_size=1000,
        lr=0.01,
        momentum=0.5,
        optimizer='sgd',
        weight_decay=0.0,
        log_interval=10,
        save_dir='models',
        data_dir='data',
        augment=False,
        visualize=False,
        val_ratio=0.1,
        no_cuda=True,
        seed=42
    )
    
    mock_device.return_value = 'cpu'
    mock_get_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_get_classes.return_value = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = (95.5, 'best_model.pt')
    mock_trainer_instance.test.return_value = (0.1, 95.0)
    mock_trainer.return_value = mock_trainer_instance
    
    # 设置模型加载返回
    mock_torch_load.return_value = {}
    
    # 执行函数
    test_acc, best_model_path = train_and_evaluate(args)
    
    # 验证结果
    assert test_acc == 95.0
    assert best_model_path == 'best_model.pt'
    mock_makedirs.assert_called_once_with('models', exist_ok=True)
    mock_get_loaders.assert_called_once()
    mock_get_model.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once_with(1)
    mock_trainer_instance.test.assert_called_once()
    

@patch('main.train_and_evaluate')
@patch('main.parse_args')
def test_main(mock_parse_args, mock_train_and_evaluate):
    """测试主函数"""
    from main import main
    
    # 设置mock对象
    args = argparse.Namespace(
        dataset='mnist',
        model='simple',
        epochs=1,
        batch_size=64,
        test_batch_size=1000,
        lr=0.01,
        momentum=0.5,
        optimizer='sgd',
        weight_decay=0.0,
        log_interval=10,
        save_dir='models',
        data_dir='data',
        augment=False,
        visualize=False,
        val_ratio=0.1,
        no_cuda=True,
        seed=42
    )
    mock_parse_args.return_value = args
    mock_train_and_evaluate.return_value = (95.0, 'best_model.pt')
    
    # 执行主函数
    main()
    
    # 验证函数调用
    mock_parse_args.assert_called_once()
    mock_train_and_evaluate.assert_called_once_with(args) 