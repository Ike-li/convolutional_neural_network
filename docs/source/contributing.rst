贡献指南
========

欢迎贡献
--------

我们欢迎各种形式的贡献，包括但不限于：

* 报告问题
* 提交改进建议
* 添加新功能
* 改进文档
* 修复bug
* 优化性能

开发环境设置
-----------

1. 克隆仓库：

   .. code-block:: bash

      git clone https://github.com/yourusername/cnn.git
      cd cnn

2. 创建虚拟环境：

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      # 或
      venv\Scripts\activate  # Windows

3. 安装开发依赖：

   .. code-block:: bash

      pip install -e ".[dev]"

代码风格
-------

我们使用以下工具来保持代码风格的一致性：

* Black: 代码格式化
* isort: import语句排序
* flake8: 代码检查
* mypy: 类型检查

在提交代码前，请运行以下命令：

.. code-block:: bash

   # 格式化代码
   black .
   isort .

   # 运行检查
   flake8
   mypy .

提交代码
-------

1. 创建新分支：

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. 提交更改：

   .. code-block:: bash

      git add .
      git commit -m "描述你的更改"

3. 推送到远程仓库：

   .. code-block:: bash

      git push origin feature/your-feature-name

4. 创建Pull Request

提交信息规范
-----------

提交信息应遵循以下格式：

.. code-block:: text

   <类型>: <描述>

   [可选的详细描述]

   [可选的关闭问题引用]

类型包括：

* feat: 新功能
* fix: bug修复
* docs: 文档更改
* style: 代码风格更改（不影响代码运行）
* refactor: 代码重构
* perf: 性能优化
* test: 测试相关
* build: 构建系统或外部依赖更改
* ci: CI配置更改

示例：

.. code-block:: text

   feat: 添加ResNet模型支持

   添加了ResNet-18和ResNet-50模型实现，包括：
   - 基础残差块
   - 瓶颈残差块
   - 完整的ResNet架构

   Closes #123

测试
----

在提交代码前，请确保所有测试通过：

.. code-block:: bash

   pytest

添加新测试：

.. code-block:: python

   def test_new_feature():
       # 测试代码
       assert result == expected

文档
----

* 所有新功能必须包含文档
* 使用Google风格的文档字符串
* 更新相关文档页面
* 生成并检查文档

生成文档：

.. code-block:: bash

   cd docs
   make html

检查文档：

.. code-block:: bash

   cd docs/build/html
   python -m http.server
   # 在浏览器中打开 http://localhost:8000

问题报告
-------

报告问题时，请包含：

* 问题描述
* 复现步骤
* 期望行为
* 实际行为
* 环境信息
* 相关代码
* 错误信息（如果有）

示例：

.. code-block:: text

   标题：训练时出现内存错误

   描述：
   使用batch_size=128训练模型时出现内存不足错误。

   复现步骤：
   1. 安装依赖
   2. 运行训练脚本
   3. 设置batch_size=128

   期望行为：
   训练正常进行

   实际行为：
   出现CUDA内存不足错误

   环境信息：
   - Python 3.8
   - PyTorch 1.7.0
   - CUDA 11.0
   - GPU: NVIDIA RTX 3080

   相关代码：
   ```python
   trainer.train(batch_size=128)
   ```

   错误信息：
   RuntimeError: CUDA out of memory

行为准则
-------

* 尊重所有贡献者
* 接受建设性批评
* 关注问题本身
* 保持专业和友善
* 帮助维护社区氛围

联系方式
-------

* GitHub Issues: https://github.com/yourusername/cnn/issues
* 电子邮件: your.email@example.com
* 讨论区: https://github.com/yourusername/cnn/discussions