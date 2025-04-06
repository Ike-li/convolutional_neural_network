安装指南
========

系统要求
--------

* Python 3.7 或更高版本
* PyTorch 1.7.0 或更高版本
* CUDA（可选，用于GPU加速）

安装步骤
--------

1. 克隆仓库：

   .. code-block:: bash

      git clone https://github.com/yourusername/cnn.git
      cd cnn

2. 创建虚拟环境（推荐）：

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      # 或
      venv\Scripts\activate  # Windows

3. 安装依赖：

   .. code-block:: bash

      pip install -e .

   这将安装所有必需的依赖项。

4. 安装开发依赖（可选）：

   .. code-block:: bash

      pip install -e ".[dev]"

   这将安装用于开发和测试的额外依赖项。

5. 安装文档依赖（可选）：

   .. code-block:: bash

      pip install -e ".[docs]"

   这将安装用于生成文档的额外依赖项。

验证安装
--------

安装完成后，可以通过以下命令验证安装是否成功：

.. code-block:: bash

   python -c "import cnn; print(cnn.__version__)"

如果安装成功，将显示当前版本号。

常见问题
--------

1. CUDA相关错误
   ~~~~~~~~~~~~~

   如果遇到CUDA相关错误，请确保：

   * 已安装正确版本的CUDA
   * PyTorch版本与CUDA版本兼容
   * 使用正确的PyTorch安装命令（例如：从PyTorch官网获取适合您系统的安装命令）

2. 依赖冲突
   ~~~~~~~~~

   如果遇到依赖冲突，建议：

   * 使用虚拟环境
   * 确保所有依赖版本兼容
   * 尝试更新或降级特定包

3. 内存不足
   ~~~~~~~~~

   如果遇到内存不足错误，可以：

   * 减小批次大小
   * 使用较小的模型
   * 启用梯度累积
   * 使用混合精度训练

获取帮助
--------

如果您在安装过程中遇到问题，请：

1. 查看 `GitHub Issues <https://github.com/yourusername/cnn/issues>`_
2. 提交新的 Issue
3. 发送邮件至 your.email@example.com