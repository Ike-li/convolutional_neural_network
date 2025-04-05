"""
CNN项目示例脚本

提供几个常用的示例命令，方便快速开始使用
"""

import argparse
import subprocess


def run_mnist_simple():
    """运行简单CNN模型训练MNIST数据集"""
    command = "python main.py --dataset mnist --model simple --epochs 5 --visualize"
    print(f"运行命令: {command}")
    subprocess.run(command, shell=True)


def run_fashion_mnist_simple():
    """运行简单CNN模型训练Fashion-MNIST数据集"""
    command = (
        "python main.py --dataset fashion_mnist --model simple --epochs 5 --visualize"
    )
    print(f"运行命令: {command}")
    subprocess.run(command, shell=True)


def run_mnist_deep():
    """运行深层CNN模型训练MNIST数据集"""
    command = "python main.py --dataset mnist --model deep --epochs 10 --visualize"
    print(f"运行命令: {command}")
    subprocess.run(command, shell=True)


def run_fashion_mnist_deep():
    """运行深层CNN模型训练Fashion-MNIST数据集"""
    command = (
        "python main.py --dataset fashion_mnist --model deep --epochs 10 --visualize"
    )
    print(f"运行命令: {command}")
    subprocess.run(command, shell=True)


def main():
    parser = argparse.ArgumentParser(description="CNN项目示例脚本")
    parser.add_argument(
        "example",
        choices=[
            "mnist_simple",
            "fashion_mnist_simple",
            "mnist_deep",
            "fashion_mnist_deep",
        ],
        help="要运行的示例",
    )

    args = parser.parse_args()

    if args.example == "mnist_simple":
        run_mnist_simple()
    elif args.example == "fashion_mnist_simple":
        run_fashion_mnist_simple()
    elif args.example == "mnist_deep":
        run_mnist_deep()
    elif args.example == "fashion_mnist_deep":
        run_fashion_mnist_deep()


if __name__ == "__main__":
    main()
