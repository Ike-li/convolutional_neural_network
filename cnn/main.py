"""
主程序入口

用于训练和评估CNN模型
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import ModelFactory
from .data.datasets import DatasetFactory
from .training.trainer import Trainer
from .config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_SAVE_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_FILE,
    USE_CUDA,
    SEED,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CNN训练程序")
    parser.add_argument(
        "--model", type=str, required=True, help="模型名称 (simple_cnn/deep_cnn)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="数据集名称 (mnist/fashion_mnist)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批次大小"
    )
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE, help="学习率"
    )
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM, help="动量")
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="权重衰减"
    )
    parser.add_argument(
        "--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, help="日志间隔"
    )
    parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="模型保存目录"
    )
    parser.add_argument("--no-cuda", action="store_true", help="禁用CUDA")
    return parser.parse_args()


def setup_logging(log_level: str = DEFAULT_LOG_LEVEL) -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=DEFAULT_LOG_FORMAT,
        handlers=[logging.FileHandler(DEFAULT_LOG_FILE), logging.StreamHandler()],
    )


def main() -> None:
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("开始训练")
    logger.info(f"参数: {args}")

    # 设置设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda and USE_CUDA else "cpu"
    )
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据集
    dataset_factory = DatasetFactory()
    train_dataset = dataset_factory.create(args.dataset, train=True)
    test_dataset = dataset_factory.create(args.dataset, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 创建模型
    model_factory = ModelFactory()
    model = model_factory.create(args.model)
    model = model.to(device)

    # 创建优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_interval=args.log_interval,
    )

    # 训练模型
    trainer.train(epochs=args.epochs)

    # 保存模型
    model_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到: {model_path}")

    # 评估模型
    test_loss, test_acc = trainer.evaluate()
    logger.info(f"测试集损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
