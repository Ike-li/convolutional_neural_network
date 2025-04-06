示例
====

基础示例
--------

MNIST分类
~~~~~~~~~

使用简单CNN模型对MNIST数据集进行分类：

.. code-block:: python

   import torch
   from cnn.models import SimpleCNN
   from cnn.data import MNISTDataset
   from cnn.training import Trainer
   from cnn.utils import Logger, Visualizer

   # 设置设备
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # 创建数据集
   train_dataset = MNISTDataset(train=True)
   test_dataset = MNISTDataset(train=False)

   # 创建模型
   model = SimpleCNN().to(device)

   # 创建训练器
   trainer = Trainer(
       model=model,
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       device=device,
       batch_size=32,
       learning_rate=0.001
   )

   # 创建日志记录器和可视化器
   logger = Logger('logs/mnist_example.log')
   visualizer = Visualizer('plots/mnist_example')

   # 训练模型
   trainer.train(epochs=10, logger=logger, visualizer=visualizer)

   # 评估模型
   test_metrics = trainer.evaluate()
   logger.log_test_results(test_metrics)

Fashion-MNIST分类
~~~~~~~~~~~~~~~

使用深度CNN模型对Fashion-MNIST数据集进行分类：

.. code-block:: python

   import torch
   from cnn.models import DeepCNN
   from cnn.data import FashionMNISTDataset
   from cnn.training import Trainer
   from cnn.utils import Logger, Visualizer

   # 设置设备
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # 创建数据集
   train_dataset = FashionMNISTDataset(train=True, augment=True)
   test_dataset = FashionMNISTDataset(train=False)

   # 创建模型
   model = DeepCNN().to(device)

   # 创建训练器
   trainer = Trainer(
       model=model,
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       device=device,
       batch_size=64,
       learning_rate=0.0001
   )

   # 创建日志记录器和可视化器
   logger = Logger('logs/fashion_mnist_example.log')
   visualizer = Visualizer('plots/fashion_mnist_example')

   # 训练模型
   trainer.train(
       epochs=50,
       logger=logger,
       visualizer=visualizer,
       early_stopping=True,
       patience=5
   )

   # 评估模型
   test_metrics = trainer.evaluate()
   logger.log_test_results(test_metrics)

高级示例
--------

数据增强
~~~~~~~~

使用多种数据增强方法：

.. code-block:: python

   from cnn.data import MNISTDataset
   from torchvision import transforms

   # 定义数据增强转换
   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])

   # 创建数据集
   train_dataset = MNISTDataset(
       train=True,
       transform=transform
   )

模型检查点
~~~~~~~~~

使用模型检查点保存和加载：

.. code-block:: python

   from cnn.training import Trainer, ModelCheckpoint

   # 创建检查点回调
   checkpoint = ModelCheckpoint(
       save_dir='checkpoints',
       save_freq=5,
       monitor='val_loss',
       mode='min'
   )

   # 创建训练器
   trainer = Trainer(
       model=model,
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       callbacks=[checkpoint]
   )

   # 训练模型
   trainer.train(epochs=50)

   # 加载最佳模型
   best_model = trainer.load_checkpoint('checkpoints/best_model.pth')

TensorBoard可视化
~~~~~~~~~~~~~~

使用TensorBoard监控训练过程：

.. code-block:: python

   from cnn.training import Trainer, TensorBoardCallback

   # 创建TensorBoard回调
   tensorboard = TensorBoardCallback(
       log_dir='runs/experiment1',
       log_freq=100
   )

   # 创建训练器
   trainer = Trainer(
       model=model,
       train_dataset=train_dataset,
       test_dataset=test_dataset,
       callbacks=[tensorboard]
   )

   # 训练模型
   trainer.train(epochs=50)

自定义模型
~~~~~~~~~

创建自定义CNN模型：

.. code-block:: python

   import torch.nn as nn
   from cnn.models import BaseModel

   class CustomCNN(BaseModel):
       def __init__(self, num_classes=10):
           super().__init__()
           self.features = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2)
           )
           self.classifier = nn.Sequential(
               nn.Linear(128 * 3 * 3, 512),
               nn.ReLU(),
               nn.Dropout(0.5),
               nn.Linear(512, num_classes)
           )

       def forward(self, x):
           x = self.features(x)
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
           return x

   # 使用自定义模型
   model = CustomCNN().to(device)
   trainer = Trainer(
       model=model,
       train_dataset=train_dataset,
       test_dataset=test_dataset
   )
   trainer.train(epochs=50)

自定义数据集
~~~~~~~~~~

创建自定义数据集：

.. code-block:: python

   from torch.utils.data import Dataset
   from PIL import Image
   import os

   class CustomDataset(Dataset):
       def __init__(self, root_dir, transform=None):
           self.root_dir = root_dir
           self.transform = transform
           self.classes = sorted(os.listdir(root_dir))
           self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
           self.samples = []
           for cls in self.classes:
               cls_dir = os.path.join(root_dir, cls)
               for img_name in os.listdir(cls_dir):
                   self.samples.append((
                       os.path.join(cls_dir, img_name),
                       self.class_to_idx[cls]
                   ))

       def __len__(self):
           return len(self.samples)

       def __getitem__(self, idx):
           img_path, label = self.samples[idx]
           image = Image.open(img_path).convert('RGB')
           if self.transform:
               image = self.transform(image)
           return image, label

   # 使用自定义数据集
   train_dataset = CustomDataset(
       root_dir='data/train',
       transform=transforms.Compose([
           transforms.Resize((28, 28)),
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
       ])
   )