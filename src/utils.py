import torch
import torch.nn as nn
from torchvision import datasets, transforms


def check_feature_map_sizes(model, input_size=(1, 28, 28), device="cpu"):
    """
    检查卷积神经网络模型的特征图尺寸，帮助调试

    Args:
        model: 要检查的CNN模型
        input_size: 输入大小，默认为MNIST/Fashion-MNIST的(1, 28, 28)
        device: 计算设备
    """
    # 创建一个样例输入
    x = torch.randn(1, *input_size).to(device)

    # 注册钩子函数
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 获取所有卷积层和池化层
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            layer.register_forward_hook(get_activation(name))

    # 执行前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)

    # 打印各层的输出尺寸
    print(f"输入尺寸: {x.shape}")
    print("\n各层的输出尺寸:")
    for name, feature in sorted(activation.items()):
        print(f"{name}: {feature.shape}")

    # 最终输出尺寸
    print(f"\n最终输出尺寸: {output.shape}")


def detailed_model_analysis(model, input_size=(1, 28, 28), device="cpu"):
    """
    对模型进行详细分析，输出每一层的参数和特征图尺寸

    Args:
        model: 要分析的CNN模型
        input_size: 输入大小，默认为MNIST/Fashion-MNIST的(1, 28, 28)
        device: 计算设备

    Returns:
        None，但打印出详细的模型分析信息
    """
    print(f"\n{'=' * 30} 模型详细分析 {'=' * 30}")
    print(f"模型类型: {model.__class__.__name__}")

    # 打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 创建样例输入
    x = torch.randn(1, *input_size).to(device)

    # 存储每一层的输出
    layer_outputs = {}
    hooks = []

    # 为每一层注册钩子
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):

            def hook_fn(name):
                def hook(module, input, output):
                    layer_outputs[name] = output

                return hook

            handle = layer.register_forward_hook(hook_fn(name))
            hooks.append(handle)

    # 执行前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)

    # 移除钩子
    for handle in hooks:
        handle.remove()

    # 打印详细信息
    print(f"\n输入形状: {x.shape}")
    print(f"\n{'=' * 70}")
    print(f"{'层名称':<25} {'类型':<20} {'输出形状':<20} {'参数量':<10}")
    print(f"{'-' * 70}")

    # 打印每一层的信息
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d)):
            layer_type = layer.__class__.__name__

            # 获取参数量
            params = sum(p.numel() for p in layer.parameters())

            # 获取输出形状
            output_shape = "N/A"
            if name in layer_outputs:
                output_shape = str(tuple(layer_outputs[name].shape))

            print(f"{name:<25} {layer_type:<20} {output_shape:<20} {params:<10,}")

    print(f"{'=' * 70}")
    print(f"最终输出形状: {output.shape}")
    print(f"总参数量: {total_params:,}")
    print(f"{'=' * 70}\n")

    return layer_outputs


def check_model_compatibility():
    """
    检查模型与数据加载器的兼容性，确保特征图尺寸计算正确
    """
    from models import SimpleCNN, DeepCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据加载器获取一个样本
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    sample_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    sample_data, _ = next(iter(sample_loader))

    # 检查SimpleCNN
    print("检查SimpleCNN模型:")
    model = SimpleCNN().to(device)
    try:
        # 使用新的详细分析函数
        detailed_model_analysis(model, device=device)
    except Exception as e:
        print(f"模型分析失败: {str(e)}")
        # 回退到简单的特征图尺寸检查
        check_feature_map_sizes(model, device=device)

    # 测试一个完整的前向传播
    try:
        with torch.no_grad():
            _ = model(sample_data.to(device))
        print("SimpleCNN前向传播成功!")
    except Exception as e:
        print(f"SimpleCNN前向传播失败: {str(e)}")

    # 检查DeepCNN
    print("\n检查DeepCNN模型:")
    model = DeepCNN().to(device)
    try:
        # 使用新的详细分析函数
        detailed_model_analysis(model, device=device)
    except Exception as e:
        print(f"模型分析失败: {str(e)}")
        # 回退到简单的特征图尺寸检查
        check_feature_map_sizes(model, device=device)

    # 测试一个完整的前向传播
    try:
        with torch.no_grad():
            model(sample_data.to(device))
        print("DeepCNN前向传播成功!")
    except Exception as e:
        print(f"DeepCNN前向传播失败: {str(e)}")


if __name__ == "__main__":
    check_model_compatibility()
