"""
主程序模块

本模块提供训练和评估模型的命令行接口。
"""

import logging
import argparse
from pathlib import Path

import torch

from .config import Config, ModelConfig, TrainingConfig, DataConfig
from .models import get_model
from .data_loader import get_data_loaders, get_dataset_classes
from .trainer import Trainer
from .visualization import plot_images, plot_confusion_matrix


def setup_logging(log_dir: str = "logs") -> None:
    """
    设置日志记录

    Args:
        log_dir: 日志目录
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CNN模型训练和评估")

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "deep"],
        help="模型类型：'simple'或'deep'",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="分类数量")
    parser.add_argument("--use-residual", action="store_true", help="是否使用残差连接")
    parser.add_argument(
        "--use-attention", action="store_true", help="是否使用注意力机制"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "silu", "gelu"],
        help="激活函数类型",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout比率")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=64, help="训练批次大小")
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, help="测试批次大小"
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--momentum", type=float, default=0.5, help="动量")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="优化器类型",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "none"],
        help="学习率调度器类型",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10, help="早停耐心值"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="是否使用混合精度训练"
    )
    parser.add_argument("--log-interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save-dir", type=str, default="models", help="模型保存目录")

    # 数据参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="数据集名称",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="数据存储目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--augment", action="store_true", help="是否使用数据增强")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--pin-memory", action="store_true", help="是否使用固定内存")
    parser.add_argument("--cache-data", action="store_true", help="是否缓存数据")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="训练设备"
    )
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式")

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> Config:
    """
    从命令行参数创建配置对象

    Args:
        args: 命令行参数

    Returns:
        配置对象
    """
    model_config = ModelConfig(
        name=args.model,
        num_classes=args.num_classes,
        use_residual=args.use_residual,
        use_attention=args.use_attention,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )

    data_config = DataConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        augment=args.augment,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        cache_data=args.cache_data,
    )

    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        seed=args.seed,
        device=args.device,
        debug=args.debug,
    )


def set_seed(seed: int) -> None:
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志
    setup_logging()
    logger = logging.getLogger("CNN.main")

    # 创建配置
    config = create_config(args)

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = torch.device(config.device)
    logger.info(f"使用设备: {device}")

    # 获取数据加载器
    logger.info(f"加载数据集: {config.data.dataset}")
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 获取类别名称
    classes = get_dataset_classes(config.data.dataset)

    # 创建模型
    logger.info(f"创建模型: {config.model.name}")
    model = get_model(config.model)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )

    # 训练模型
    logger.info("开始训练")
    best_val_acc, best_model_path = trainer.train(config.training.epochs)
    logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    logger.info(f"最佳模型保存在: {best_model_path}")

    # 绘制训练历史
    plot_path = Path(config.training.save_dir) / "training_history.png"
    trainer.plot_training_history(plot_path)
    logger.info(f"训练历史图表保存在: {plot_path}")

    # 获取训练摘要
    summary = trainer.get_training_summary()
    logger.info("训练摘要:")
    logger.info(f"- 最佳验证准确率: {summary['best_val_acc']:.2f}%")
    logger.info(f"- 最佳模型轮数: {summary['best_epoch']}")
    logger.info(f"- 训练时间: {summary['training_time']:.1f}秒")

    # 可视化
    if config.debug:
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

            logger.info("可视化完成!")
        except Exception as e:
            logger.error(f"可视化过程中出错: {str(e)}")


if __name__ == "__main__":
    main()
