# 卷积神经网络项目

这是一个用于图像分类的卷积神经网络（CNN）实现项目。该项目支持MNIST和Fashion-MNIST数据集，提供了简单和深层两种CNN模型架构，并包含多种现代化的优化技术。

## 特性

- 支持多种模型架构：
  - SimpleCNN：适用于简单任务的基础CNN模型
  - DeepCNN：包含更多层和现代化组件的深层模型

- 现代化组件支持：
  - 残差连接（ResNet风格）
  - 注意力机制
  - 批量归一化
  - 多种激活函数（ReLU、SiLU/Swish、GELU）

- 训练优化：
  - 混合精度训练
  - 梯度累积
  - 早停机制
  - 学习率调度
  - 数据增强

- 数据集支持：
  - MNIST
  - Fashion-MNIST

## 安装

### 使用 Poetry（推荐）

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/convolutional_neural_network.git
cd convolutional_neural_network
```

2. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装依赖：
```bash
poetry install
```

4. 激活虚拟环境：
```bash
poetry shell
```

### 使用 pip

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/convolutional_neural_network.git
cd convolutional_neural_network
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练

训练简单的CNN模型：
```bash
python train.py --model simple --dataset mnist
```

训练深层CNN模型：
```bash
python train.py --model deep --dataset fashion_mnist
```

### 高级选项

使用残差连接和注意力机制：
```bash
python train.py --model deep --use-residual --use-attention
```

启用混合精度训练：
```bash
python train.py --model deep --mixed-precision
```

使用数据增强：
```bash
python train.py --simple --augment
```

### 完整参数说明

#### 模型参数
- `--model`：模型类型 ['simple', 'deep']
- `--num-classes`：分类数量（默认：10）
- `--use-residual`：启用残差连接
- `--use-attention`：启用注意力机制
- `--activation`：激活函数类型 ['relu', 'silu', 'gelu']
- `--dropout-rate`：Dropout比率（默认：0.5）

#### 训练参数
- `--batch-size`：训练批次大小（默认：64）
- `--test-batch-size`：测试批次大小（默认：1000）
- `--epochs`：训练轮数（默认：50）
- `--lr`：学习率（默认：0.01）
- `--momentum`：动量（默认：0.5）
- `--weight-decay`：权重衰减（默认：1e-4）
- `--optimizer`：优化器类型 ['sgd', 'adam']
- `--scheduler`：学习率调度器 ['plateau', 'cosine', 'step', 'none']
- `--early-stopping-patience`：早停耐心值（默认：10）
- `--gradient-accumulation-steps`：梯度累积步数（默认：1）
- `--mixed-precision`：启用混合精度训练
- `--log-interval`：日志打印间隔（默认：100）
- `--save-dir`：模型保存目录（默认：'models'）

#### 数据参数
- `--dataset`：数据集名称 ['mnist', 'fashion_mnist']
- `--data-dir`：数据存储目录（默认：'data'）
- `--val-ratio`：验证集比例（默认：0.2）
- `--augment`：启用数据增强
- `--num-workers`：数据加载线程数（默认：4）
- `--pin-memory`：启用内存固定
- `--cache-data`：启用数据缓存

#### 其他参数
- `--seed`：随机种子（默认：42）
- `--device`：训练设备 ['cuda', 'cpu']
- `--debug`：启用调试模式（包含可视化）

## 项目结构

```
convolutional_neural_network/
├── src/                    # 源代码目录
│   ├── main.py            # 主要实现逻辑
│   ├── models.py          # 模型定义
│   ├── data_loader.py     # 数据加载
│   ├── trainer.py         # 训练器
│   ├── visualization.py   # 可视化工具
│   ├── utils.py           # 工具函数
│   ├── config.py          # 配置管理
│   └── unitests/          # 单元测试
├── data/                   # 数据目录
├── models/                 # 模型保存目录
├── logs/                   # 日志目录
├── docs/                   # 文档目录
├── train.py               # 训练入口脚本
├── requirements.txt       # pip 依赖
├── pyproject.toml         # Poetry 配置
├── poetry.lock           # Poetry 依赖锁定
└── README.md             # 项目文档
```

## 测试

项目使用 pytest 进行测试。运行测试：

```bash
# 使用 Poetry
poetry run pytest

# 使用 pip
pytest
```

运行测试覆盖率报告：

```bash
# 使用 Poetry
poetry run pytest --cov=src

# 使用 pip
pytest --cov=src
```

## 训练结果

训练过程会生成以下输出：

1. 模型文件：保存在 `models/` 目录下
2. 训练日志：保存在 `logs/training.log`
3. 可视化结果（调试模式）：
   - 训练历史图表
   - 样本预测结果
   - 混淆矩阵

## 性能基准

在MNIST数据集上的典型性能：

- SimpleCNN：
  - 准确率：~98.5%
  - 训练时间：~5分钟（GPU）
  - 模型大小：~2.5MB

- DeepCNN：
  - 准确率：~99.2%
  - 训练时间：~10分钟（GPU）
  - 模型大小：~15MB

在Fashion-MNIST数据集上的典型性能：

- SimpleCNN：
  - 准确率：~89.5%
  - 训练时间：~5分钟（GPU）
  - 模型大小：~2.5MB

- DeepCNN：
  - 准确率：~93.2%
  - 训练时间：~10分钟（GPU）
  - 模型大小：~15MB

## 贡献

欢迎提交问题和拉取请求。对于重大更改，请先开issue讨论您想要更改的内容。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
