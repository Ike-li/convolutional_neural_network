# 卷积神经网络项目

基于PyTorch实现的卷积神经网络，用于MNIST和Fashion-MNIST数据集的图像分类。

## 项目结构

```
.
├── cnn/                # 主包目录
│   ├── data/           # 数据加载相关模块
│   │   ├── __init__.py
│   │   └── loader.py   # 数据集加载器
│   ├── models/         # 模型相关模块
│   │   ├── __init__.py
│   │   ├── cnn_models.py  # CNN模型定义
│   │   └── trainer.py  # 模型训练器
│   ├── utils/          # 工具模块
│   │   ├── __init__.py
│   │   └── model_utils.py # 工具函数和调试模块
│   ├── visualization/  # 可视化模块
│   │   ├── __init__.py
│   │   └── plots.py    # 可视化工具
│   ├── __init__.py
│   ├── main.py         # 主程序
│   └── verify_model.py # 模型验证脚本
├── data/               # 数据集目录（自动创建）
├── examples/           # 示例目录
│   ├── README.md       # 示例说明
│   └── basic_examples.py # 基本使用示例
├── models/             # 保存的模型（训练后生成）
├── tests/              # 测试目录
│   ├── __init__.py
│   ├── test_models.py  # 模型测试
│   └── test_data_loader.py # 数据加载器测试
├── train.py            # 训练入口脚本
├── validate.py         # 验证入口脚本
├── pyproject.toml      # 项目依赖配置
└── README.md           # 项目说明
```

## 特性

- 支持MNIST和Fashion-MNIST数据集
- 提供简单CNN和深层CNN两种模型架构
- 包含完整的训练、验证和测试流程
- 支持模型性能可视化（混淆矩阵、过滤器、特征图等）
- 训练过程可视化（损失曲线、准确率曲线）
- 提供模型结构验证和调试工具
- 标准的Python包结构，便于安装和分发
- 自动化测试
- 命令行入口点

## 环境要求

- Python 3.10+
- PyTorch 2.6.0+
- torchvision 0.21.0+
- NumPy 2.2.0+
- Matplotlib 3.10.0+
- scikit-learn 1.6.0+
- tqdm 4.66.0+
- typer 0.9.0+
- rich 13.7.0+

## 安装

```bash
# 使用Poetry安装依赖
poetry install

# 或者使用pip
pip install -e .
```

## 项目重组

如果您想将现有代码重组为标准的Python包结构，可以运行以下命令：

```bash
# 重组项目结构
python src/restructure.py
```

这将:
1. 创建符合Python最佳实践的包结构
2. 将现有代码复制到相应的模块中
3. 创建入口点脚本
4. 设置基本的测试框架

**注意**：原始文件会保留，这只是复制操作。您需要检查新文件并调整导入语句以适应新的包结构。

## 使用方法

### 验证模型

在训练前首先验证模型结构是否正确：

```bash
# 使用入口脚本
python validate.py --model simple --dataset mnist

# 或直接使用模块
python -m cnn.verify_model --model simple --dataset mnist
```

### 训练模型

```bash
# 使用入口脚本
python train.py --dataset mnist --model simple --epochs 10

# 使用模块
python -m cnn.main --dataset mnist --model simple --epochs 10

# 启用可视化
python train.py --dataset mnist --model simple --epochs 10 --visualize

# 使用数据增强
python train.py --dataset fashion_mnist --model deep --epochs 20 --augment
```

### 命令行参数

- `--dataset`: 数据集选择 (mnist, fashion_mnist)
- `--model`: 模型选择 (simple, deep)
- `--batch-size`: 训练批次大小
- `--test-batch-size`: 测试批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--momentum`: SGD动量
- `--weight-decay`: 权重衰减
- `--optimizer`: 优化器选择 (sgd, adam)
- `--augment`: 使用数据增强
- `--val-ratio`: 验证集比例
- `--no-cuda`: 禁用CUDA训练
- `--seed`: 随机种子
- `--log-interval`: 训练日志打印间隔
- `--save-dir`: 模型保存目录
- `--visualize`: 启用可视化
- `--data-dir`: 数据存储目录

## 模型架构

### 简单CNN

一个基础的卷积神经网络，包含以下层：
- 两个卷积层（带padding=1保持特征图尺寸）
- 一个最大池化层（在第二次卷积后应用）
- Dropout层
- 两个全连接层

特征图尺寸变化：
- 输入: [batch, 1, 28, 28]
- 第一次卷积: [batch, 32, 28, 28]
- 第二次卷积: [batch, 64, 28, 28]
- 池化后: [batch, 64, 14, 14]
- 展平后: [batch, 64*14*14=12544]

### 深层CNN

一个更深的卷积神经网络，包含以下层：
- 三个卷积层，每个卷积层后接批量归一化
- 三个最大池化层（每个卷积层后）
- Dropout层
- 两个全连接层

特征图尺寸变化：
- 输入: [batch, 1, 28, 28]
- 第一次池化后: [batch, 32, 14, 14]
- 第二次池化后: [batch, 64, 7, 7]
- 第三次池化后: [batch, 128, 3, 3]
- 展平后: [batch, 128*3*3=1152]

## 测试

项目包含自动化测试，确保模型和数据加载器的正确性：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_models.py
```

## 开发工具

项目配置了以下开发工具：

- **black**: 代码格式化
- **isort**: 导入语句排序
- **mypy**: 类型检查
- **flake8**: 代码质量检查

## 示例

查看`examples`目录中的示例：

```bash
# 在MNIST数据集上训练简单CNN模型
python examples/basic_examples.py mnist_simple

# 在Fashion-MNIST数据集上训练深层CNN模型
python examples/basic_examples.py fashion_mnist_deep
```

## 性能

- MNIST数据集上，简单CNN模型通常可以达到约98%的准确率
- Fashion-MNIST数据集上，深层CNN模型通常可以达到约92%的准确率

## 常见问题解决

如果遇到"mat1 and mat2 shapes cannot be multiplied"等维度不匹配错误，可以：

1. 使用验证脚本检查模型：`python validate.py --model simple`
2. 使用utils中的调试工具分析模型特征图尺寸
3. 检查模型定义中全连接层的输入维度是否与展平后的特征图尺寸匹配
