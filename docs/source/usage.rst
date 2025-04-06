使用指南
========

基本用法
--------

训练模型
~~~~~~~~

使用以下命令训练模型：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --epochs 10 --batch_size 32

主要参数说明：

* ``--model``: 模型架构（simple_cnn 或 deep_cnn）
* ``--dataset``: 数据集（mnist 或 fashion_mnist）
* ``--epochs``: 训练轮数
* ``--batch_size``: 批次大小
* ``--learning_rate``: 学习率
* ``--device``: 训练设备（cpu 或 cuda）

评估模型
~~~~~~~~

使用以下命令评估模型：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --mode test --checkpoint path/to/checkpoint.pth

主要参数说明：

* ``--mode``: 运行模式（train 或 test）
* ``--checkpoint``: 模型检查点文件路径

高级用法
--------

数据增强
~~~~~~~~

启用数据增强：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --augment

可用的数据增强方法：

* 随机水平翻转
* 随机旋转
* 随机裁剪
* 颜色抖动（仅适用于彩色图像）

早停
~~~~

启用早停：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --early_stopping --patience 5

参数说明：

* ``--early_stopping``: 启用早停
* ``--patience``: 容忍轮数
* ``--min_delta``: 最小改善阈值

模型检查点
~~~~~~~~~~

启用模型检查点：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --checkpoint_dir checkpoints --save_freq 5

参数说明：

* ``--checkpoint_dir``: 检查点保存目录
* ``--save_freq``: 保存频率（轮数）

TensorBoard可视化
~~~~~~~~~~~~~~~

启用TensorBoard：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --tensorboard

然后在另一个终端中运行：

.. code-block:: bash

   tensorboard --logdir runs

日志记录
~~~~~~~~

配置日志记录：

.. code-block:: bash

   python -m cnn.main --model simple_cnn --dataset mnist --log_dir logs --log_level INFO

参数说明：

* ``--log_dir``: 日志文件目录
* ``--log_level``: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）

示例
----

1. 使用简单CNN训练MNIST数据集：

   .. code-block:: bash

      python -m cnn.main --model simple_cnn --dataset mnist --epochs 10 --batch_size 32 --learning_rate 0.001 --device cuda

2. 使用深度CNN训练Fashion-MNIST数据集，启用数据增强和早停：

   .. code-block:: bash

      python -m cnn.main --model deep_cnn --dataset fashion_mnist --epochs 50 --batch_size 64 --learning_rate 0.0001 --device cuda --augment --early_stopping --patience 5

3. 评估保存的模型：

   .. code-block:: bash

      python -m cnn.main --model simple_cnn --dataset mnist --mode test --checkpoint checkpoints/best_model.pth

4. 使用TensorBoard监控训练过程：

   .. code-block:: bash

      python -m cnn.main --model simple_cnn --dataset mnist --tensorboard --log_dir runs/experiment1

常见问题
--------

1. 内存不足
   ~~~~~~~~~

   如果遇到内存不足错误，可以：

   * 减小批次大小（--batch_size）
   * 使用较小的模型（--model simple_cnn）
   * 减少数据增强方法
   * 使用混合精度训练（--amp）

2. 训练不收敛
   ~~~~~~~~~~

   如果模型训练不收敛，可以：

   * 调整学习率（--learning_rate）
   * 增加训练轮数（--epochs）
   * 使用更复杂的模型（--model deep_cnn）
   * 增加数据增强

3. GPU利用率低
   ~~~~~~~~~~~

   如果GPU利用率低，可以：

   * 增加批次大小（--batch_size）
   * 减少数据预处理开销
   * 使用混合精度训练（--amp）
   * 检查数据加载器的工作进程数

获取帮助
--------

如果您在使用过程中遇到问题，请：

1. 查看 `GitHub Issues <https://github.com/yourusername/cnn/issues>`_
2. 提交新的 Issue
3. 发送邮件至 your.email@example.com