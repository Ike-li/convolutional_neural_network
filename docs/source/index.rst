欢迎使用CNN项目文档
====================

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   installation
   usage
   api
   examples
   contributing

简介
----

CNN项目是一个基于PyTorch的卷积神经网络项目，用于图像分类任务。该项目支持MNIST和Fashion-MNIST数据集，提供了简单CNN和深度CNN两种模型架构，并包含了丰富的可视化和日志记录功能。

主要特点：

* 支持MNIST和Fashion-MNIST数据集
* 提供简单CNN和深度CNN两种模型架构
* 支持数据增强
* 提供丰富的可视化功能
* 支持模型检查点保存和加载
* 支持早停机制
* 支持TensorBoard可视化
* 提供详细的日志记录

快速开始
--------

安装项目：

.. code-block:: bash

   pip install -e .

训练模型：

.. code-block:: bash

   python -m cnn.main --dataset mnist --model simple_cnn --epochs 10

更多信息请参考 :doc:`installation` 和 :doc:`usage` 章节。

索引和表格
---------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`