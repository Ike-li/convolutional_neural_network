"""
验证CNN模型运行脚本

用于快速检查模型能否正确运行
"""

import torch
import torch.nn.functional as F
import argparse
import traceback
from models import SimpleCNN, DeepCNN
from utils import check_feature_map_sizes, detailed_model_analysis
from data_loader import get_mnist_loaders, get_fashion_mnist_loaders


def verify_model(model_name="simple", dataset_name="mnist"):
    """验证模型能否正确运行"""
    print(f"验证{model_name.upper()}模型在{dataset_name.upper()}数据集上的运行")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    if model_name.lower() == "simple":
        model = SimpleCNN().to(device)
    else:
        model = DeepCNN().to(device)

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 使用详细分析功能
    print("\n进行详细模型分析:")
    try:
        detailed_model_analysis(model, device=device)
    except Exception as e:
        print(f"详细模型分析失败: {str(e)}")
        print("尝试使用基本特征图尺寸检查...")
        try:
            check_feature_map_sizes(model, device=device)
        except Exception as e:
            print(f"特征图尺寸检查也失败: {str(e)}")
            print("这表明模型结构存在问题，请检查模型定义。")
            return

    # 加载小批量数据用于测试
    print(f"\n加载{dataset_name}小批量数据进行测试...")
    try:
        if dataset_name.lower() == "mnist":
            train_loader, _, _ = get_mnist_loaders(batch_size=4, test_batch_size=4)
        else:
            train_loader, _, _ = get_fashion_mnist_loaders(
                batch_size=4, test_batch_size=4
            )
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        traceback.print_exc()
        return

    # 获取一个批次的数据
    try:
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        print(f"输入数据形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
    except Exception as e:
        print(f"获取数据批次失败: {str(e)}")
        traceback.print_exc()
        return

    # 测试前向传播
    print("\n执行前向传播测试:")
    model.eval()
    try:
        with torch.no_grad():
            # 先尝试单个样本
            single_image = images[0:1]
            print(f"单样本形状: {single_image.shape}")
            output_single = model(single_image)
            print(f"单样本输出形状: {output_single.shape}")

            # 然后尝试整个批次
            output = model(images)
            _, predicted = torch.max(output, 1)

            print(f"完整批次形状: {images.shape}")
            print(f"输出形状: {output.shape}")
            print(f"真实标签: {labels.cpu().numpy()}")
            print(f"预测标签: {predicted.cpu().numpy()}")
            print(f"预测概率: {torch.exp(output).max(1)[0].cpu().numpy()}")

            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print(f"批次准确率: {accuracy:.2f}")

        print("\n模型验证成功! 可以开始训练。")

    except Exception as e:
        print(f"\n模型验证失败! 错误: {str(e)}")
        traceback.print_exc()
        print("\n请检查模型结构或数据加载器。")

        # 尝试输出更多的错误信息
        try:
            print("\n尝试使用通道分析调试...")
            x = images[0:1]  # 取第一个样本

            # 打印每一层的形状
            if model_name.lower() == "simple":
                x1 = model.conv1(x)
                print(f"conv1输出: {x1.shape}")

                x2 = F.relu(x1)
                print(f"relu1输出: {x2.shape}")

                x3 = model.conv2(x2)
                print(f"conv2输出: {x3.shape}")

                x4 = F.relu(x3)
                print(f"relu2输出: {x4.shape}")

                x5 = model.pool(x4)
                print(f"pool输出: {x5.shape}")

                x6 = model.dropout1(x5)
                print(f"dropout1输出: {x6.shape}")

                x7 = torch.flatten(x6, 1)
                print(f"flatten输出: {x7.shape}")

                # 重要！这里显示实际的扁平化尺寸，帮助调试
                print(f"扁平化输出维度: {x7.size(1)}")
                print(f"fc1需要的输入维度: {model.fc1.in_features}")
        except Exception as inner_e:
            print(f"通道分析失败: {str(inner_e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN模型验证脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "deep"],
        help="CNN模型: simple, deep (默认: simple)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist"],
        help="数据集: mnist, fashion_mnist (默认: mnist)",
    )

    args = parser.parse_args()
    verify_model(args.model, args.dataset)
