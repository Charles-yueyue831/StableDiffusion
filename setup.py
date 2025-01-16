# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/12 12:43
# @Software : PyCharm

from setuptools import setup, find_packages

"""
packages=find_packages(): 搜索当前目录以及当前目录下的所有python包（带__init__.py的目录），并一起打成egg文件包
"""
setup(
    name='stable-diffusion',
    version='0.0.1',
    description='附带注释的Stable Diffusion代码',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
