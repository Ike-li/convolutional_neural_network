"""
CNN项目主程序入口

该模块提供命令行接口用于训练和评估CNN模型，支持不同的数据集和模型架构。
"""

import argparse
import logging
import os
import random
import sys
from typing import Tuple

import numpy as np
import torch

from models import get_model
from data_loader import get_data_loaders, get_dataset_classes
from trainer import Trainer
from visualization import (
    plot_images,
    plot_confusion_matrix,
    visualize_filters,
    visualize_feature_maps,
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CNN")


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保结果可重现

    Args:
        seed: 随机种子
    """
    logger.info(f"设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="卷积神经网络训练程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据相关参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="数据集: mnist, fashion_mnist",
    )
    data_group.add_argument("--data-dir", type=str, default="data", help="数据存储目录")
    data_group.add_argument("--batch-size", type=int, default=64, help="训练批次大小")
    data_group.add_argument(
        "--test-batch-size", type=int, default=1000, help="测试批次大小"
    )
    data_group.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    data_group.add_argument("--augment", action="store_true", help="使用数据增强")

    # 模型相关参数
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "deep"],
        help="CNN模型: simple, deep",
    )

    # 训练相关参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--epochs", type=int, default=10, help="训练轮数")
    train_group.add_argument("--lr", type=float, default=0.01, help="学习率")
    train_group.add_argument("--momentum", type=float, default=0.5, help="SGD动量")
    train_group.add_argument(
        "--weight-decay", type=float, default=1e-4, help="权重衰减"
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="优化器: sgd, adam",
    )
    train_group.add_argument(
        "--log-interval", type=int, default=100, help="训练日志打印间隔"
    )

    # 输出相关参数
    output_group = parser.add_argument_group("输出参数")
    output_group.add_argument(
        "--save-dir", type=str, default="models", help="模型保存目录"
    )
    output_group.add_argument("--visualize", action="store_true", help="启用可视化")

    # 其他参数
    misc_group = parser.add_argument_group("其他参数")
    misc_group.add_argument("--no-cuda", action="store_true", help="禁用CUDA训练")
    misc_group.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> Tuple[float, str]:
    """
    训练和评估模型

    Args:
        args: 命令行参数

    Returns:
        测试准确率和最佳模型路径

    Raises:
        Exception: 数据加载、模型创建或训练过程中可能出现的各种异常
    """
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    logger.info(f"加载{args.dataset}数据集...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_cuda=use_cuda,
            val_ratio=args.val_ratio,
            augment=args.augment,
            data_dir=args.data_dir,
        )
        classes = get_dataset_classes(args.dataset)
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

    # 创建模型
    logger.info(f"创建{args.model}模型...")
    try:
        model = get_model(args.model).to(device)
        logger.info(f"模型架构:\n{model}")
    except Exception as e:
        logger.error(f"创建模型失败: {str(e)}")
        raise

    # 训练模型
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            log_interval=args.log_interval,
            save_dir=args.save_dir,
        )

        logger.info("开始训练...")
        best_val_acc, best_model_path = trainer.train(args.epochs)
        logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

        # 绘制训练历史
        history_path = os.path.join(args.save_dir, "training_history.png")
        trainer.plot_training_history(history_path)
        logger.info(f"训练历史已保存至: {history_path}")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

    # 测试模型
    try:
        logger.info("加载最佳模型进行测试...")
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc = trainer.test()
        logger.info(f"测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        raise

    # 可视化
    if args.visualize:
        try:
            logger.info("生成可视化结果...")

            # 获取一批测试数据
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            images, labels = images.to(device), labels.to(device)

            # 进行预测
            with torch.no_grad():
                output = model(images)
                predictions = output.argmax(dim=1)

            # 显示样本图像及预测结果
            plot_images(images, labels, predictions, classes)

            # 绘制混淆矩阵
            plot_confusion_matrix(model, test_loader, device, classes)

            # 可视化卷积层过滤器
            visualize_filters(model)

            # 可视化特征图
            visualize_feature_maps(model, images, device)

            logger.info("可视化完成!")
        except Exception as e:
            logger.error(f"可视化过程中出错: {str(e)}")
            # 可视化错误不中断程序

    return test_acc, best_model_path


def main() -> None:
    """
    主函数，解析参数并执行训练和评估

    Raises:
        Exception: 程序执行期间可能出现的异常
    """
    try:
        # 解析命令行参数
        args = parse_args()

        # 训练和评估模型
        test_acc, best_model_path = train_and_evaluate(args)

        # 总结结果
        logger.info("=" * 50)
        logger.info("训练总结:")
        logger.info(f"  数据集: {args.dataset}")
        logger.info(f"  模型: {args.model}")
        logger.info(f"  训练轮数: {args.epochs}")
        logger.info(f"  最佳模型路径: {best_model_path}")
        logger.info(f"  测试准确率: {test_acc:.2f}%")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("接收到中断信号，停止训练")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
