"""
项目安装配置文件
"""

from setuptools import setup, find_packages

setup(
    name="cnn",
    version="0.1.0",
    description="一个基于PyTorch的卷积神经网络项目，用于图像分类任务",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.23.0",
        "tensorboard>=2.3.0",
        "tqdm>=4.50.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.6.0",
            "flake8>=3.9.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.2.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
