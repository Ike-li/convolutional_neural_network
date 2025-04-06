"""
Sphinx配置文件
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath("../../"))

# 项目信息
project = "CNN项目"
copyright = "2024, Your Name"
author = "Your Name"
release = "0.1.0"

# 扩展
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# 主题设置
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# 自动文档设置
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# 交叉引用设置
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
}

# 语言设置
language = "zh_CN"
