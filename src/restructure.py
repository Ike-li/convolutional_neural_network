#!/usr/bin/env python3
"""
项目结构重组脚本

将现有代码移动到符合Python最佳实践的包结构中
"""

import os
import shutil
from pathlib import Path


def create_directory_structure(base_dir: Path) -> None:
    """创建符合最佳实践的目录结构"""
    # 创建主包目录
    cnn_dir = base_dir / "cnn"
    os.makedirs(cnn_dir, exist_ok=True)

    # 创建子包目录
    for subdir in ["data", "models", "utils", "visualization"]:
        os.makedirs(cnn_dir / subdir, exist_ok=True)

    # 创建必要的初始化文件
    init_files = [
        cnn_dir / "__init__.py",
        cnn_dir / "data" / "__init__.py",
        cnn_dir / "models" / "__init__.py",
        cnn_dir / "utils" / "__init__.py",
        cnn_dir / "visualization" / "__init__.py",
    ]

    for init_file in init_files:
        with open(init_file, "w") as f:
            f.write('"""CNN package.\n\n"""\n')

    # 创建测试目录
    tests_dir = base_dir / "tests"
    os.makedirs(tests_dir, exist_ok=True)

    with open(tests_dir / "__init__.py", "w") as f:
        f.write('"""Test package.\n\n"""\n')

    # 创建示例目录
    examples_dir = base_dir / "examples"
    os.makedirs(examples_dir, exist_ok=True)


def map_files_to_new_structure(base_dir: Path) -> dict:
    """映射现有文件到新的结构"""
    file_mappings = {
        # 源文件映射
        base_dir / "models.py": base_dir / "cnn" / "models" / "cnn_models.py",
        base_dir / "data_loader.py": base_dir / "cnn" / "data" / "loader.py",
        base_dir / "trainer.py": base_dir / "cnn" / "models" / "trainer.py",
        base_dir / "visualization.py": base_dir / "cnn" / "visualization" / "plots.py",
        base_dir / "utils.py": base_dir / "cnn" / "utils" / "model_utils.py",
        base_dir / "main.py": base_dir / "cnn" / "main.py",
        base_dir / "verify_model.py": base_dir / "cnn" / "verify_model.py",
        # 示例文件
        base_dir / "examples.py": base_dir / "examples" / "basic_examples.py",
    }

    return file_mappings


def create_entry_points(base_dir: Path) -> None:
    """创建更方便的入口脚本"""
    # 训练脚本
    train_script = base_dir / "train.py"
    with open(train_script, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
训练模型的命令行入口点
\"\"\"
from cnn.main import main

if __name__ == "__main__":
    main()
""")

    # 验证脚本
    validate_script = base_dir / "validate.py"
    with open(validate_script, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
验证模型的命令行入口点
\"\"\"
from cnn.verify_model import main

if __name__ == "__main__":
    main()
""")

    # 添加可执行权限
    os.chmod(train_script, 0o755)
    os.chmod(validate_script, 0o755)


def copy_files(file_mappings: dict) -> None:
    """复制文件到新位置"""
    for src, dest in file_mappings.items():
        if src.exists():
            # 确保目标目录存在
            os.makedirs(dest.parent, exist_ok=True)

            # 复制文件
            shutil.copy2(src, dest)
            print(f"已复制: {src} -> {dest}")
        else:
            print(f"警告: 源文件不存在 {src}")


def create_test_files(base_dir: Path) -> None:
    """创建基本的测试文件"""
    tests_dir = base_dir / "tests"

    # 测试文件
    test_files = {
        "test_models.py": '''
import pytest
import torch
from cnn.models.cnn_models import SimpleCNN, DeepCNN

def test_simple_cnn_forward():
    """测试SimpleCNN的前向传播"""
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)

def test_deep_cnn_forward():
    """测试DeepCNN的前向传播"""
    model = DeepCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10)
''',
        "test_data_loader.py": '''
import pytest
from cnn.data.loader import get_dataset_classes

def test_get_dataset_classes():
    """测试获取数据集类别名称"""
    mnist_classes = get_dataset_classes("mnist")
    assert len(mnist_classes) == 10

    fashion_classes = get_dataset_classes("fashion_mnist")
    assert len(fashion_classes) == 10

    with pytest.raises(ValueError):
        get_dataset_classes("unknown_dataset")
''',
    }

    for filename, content in test_files.items():
        with open(tests_dir / filename, "w") as f:
            f.write(content)


def create_example_readme(base_dir: Path) -> None:
    """创建示例README文件"""
    examples_dir = base_dir / "examples"

    with open(examples_dir / "README.md", "w") as f:
        f.write("""# CNN项目示例

这个目录包含使用CNN项目的示例。

## 基本示例

`basic_examples.py` 提供了几个运行CNN模型的基本示例。

运行示例:

```bash
# 在MNIST数据集上训练简单CNN模型
python basic_examples.py mnist_simple

# 在Fashion-MNIST数据集上训练深层CNN模型
python basic_examples.py fashion_mnist_deep
```
""")


def main() -> None:
    """主函数"""
    base_dir = Path(__file__).parent

    print(f"开始重组项目结构: {base_dir}")

    # 1. 创建新目录结构
    create_directory_structure(base_dir)

    # 2. 映射文件
    file_mappings = map_files_to_new_structure(base_dir)

    # 3. 复制文件
    copy_files(file_mappings)

    # 4. 创建入口点
    create_entry_points(base_dir)

    # 5. 创建测试文件
    create_test_files(base_dir)

    # 6. 创建示例README
    create_example_readme(base_dir)

    print("项目结构重组完成。")
    print("请检查新文件并调整导入语句以适应新的包结构。")
    print("注意：这只是复制了文件，原始文件仍然存在。")


if __name__ == "__main__":
    main()
