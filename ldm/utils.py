# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import importlib
import math
import einops
from inspect import isfunction
import torch
import torch.nn as nn
from torch import Tensor


def default(val, d):
    """
    如果val不为None则返回val，否则返回d
    :param val: value
    :param d: value的备份
    :return:
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d


def instantiate_from_config(config):
    """

    :param config: 配置文件
                   target: ldm.models.diffusion.ddpm.LatentDiffusion
                   params:
                       linear_start: 0.00085
                       linear_end: 0.0120
    :return:
    """
    if not "target" in config:
        # 检查配置字典config中是否包含键"target"
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    """
    params = {"name": "Alice", "age": 30}
    instance = MyClass(**params)
    等价于instance = MyClass(name="Alice", age=30)
    """
    # 从ldm.models.diffusion.ddpm文件中读取LatentDiffusion类
    # 从ldm.modules.diffusion.openai_model文件中读取UNetModel类
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """
    从ldm.models.diffusion.ddpm文件中读取LatentDiffusion类
    从ldm.modules.diffusion.openai_model文件中读取UNetModel类
    :param string: 对象路径的字符串
    :param reload: 是否重新加载模块
    :return:
    """
    """
    str.rsplit(sep=None, maxsplit=-1)
        sep: 指定分割字符串时使用的分隔符
        maxsplit: 指定分割的最大次数，默认为 -1
        
        example:
            s = "a b c d e"
    
            # 不指定分隔符和最大分割次数
            print(s.rsplit())  # 输出：['a', 'b', 'c', 'd', 'e']
            
            # 指定分隔符为空格，不指定最大分割次数
            print(s.rsplit(' '))  # 输出：['a', 'b', 'c', 'd', 'e']
            
            # 指定分隔符为空格，最大分割次数为 2
            print(s.rsplit(' ', 2))  # 输出：['a b', 'c', 'd e']
            
            # 指定分隔符为空格，最大分割次数为 1
            print(s.rsplit(' ', 1))  # 输出：['a b c d', 'e']
    """
    # module: ldm.models.diffusion.ddpm
    # cls: LatentDiffusion
    module, cls = string.rsplit(".", 1)
    if reload:
        """
        importlib.import_module(name, package=None): 动态导入模块，并返回模块对象
            name: 要导入的模块名称，可以是相对名称或绝对名称
            package: 当 name 是相对名称时，package 指定父包的名称。如果是绝对名称，则 package 可以忽略或设置为 None
        
        example:
            # 动态导入 math 模块
            math_module = importlib.import_module('math')
            
            # 使用 math 模块中的函数
            result = math_module.sqrt(16)
        """
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    """
    getattr(): 从一个对象中获取一个指定的属性
    """
    # 从ldm.models.diffusion.ddpm文件中读取LatentDiffusion类
    # 从ldm.modules.diffusion.openai_model文件中读取UNetModel类
    return getattr(importlib.import_module(module, package=None), cls)


class CheckpointFunction(torch.autograd.Function):
    """
    torch.autograd.Function 是 PyTorch 中用于自定义自动求导操作的基类，允许用户自定义前向和后向传播的逻辑
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):
        """

        :param ctx: 上下文对象，用于存储前向传播过程中的信息
        :param run_function: 一个函数，它将在前向传播中被调用，用于执行某些计算操作
        :param length: 一个整数，用于将 args 划分为输入张量和输入参数
        :param args: 可变参数，包含输入张量和输入参数
        :return:
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        """
        存储当前 GPU 自动混合精度（autocast）的相关设置
            包括是否启用自动混合精度（enabled）、自动混合精度的 dtype 以及是否启用缓存（cache_enabled）
        """
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        """
        使用 torch.no_grad() 上下文管理器，在前向传播中关闭梯度计算
        """
        with torch.no_grad():
            """
            *ctx.input_tensors:
                *: 解包操作符，用于将列表或元组中的元素解包为多个参数
            """
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """

        :param ctx: 前向传播时存储信息的上下文对象
        :param output_grads: 后向传播的输出梯度，对应于 forward 方法中返回的 output_tensors 的梯度
                             output_grads的初始值为1
        :return:
        """

        """
        x.detach().requires_grad_(True):
            x.detach(): 将张量从计算图中分离，防止在前向传播中梯度的累积
            requires_grad_(True): 将分离后的张量重新设置为需要计算梯度，以便在后向传播中计算梯度
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

        """
        torch.enable_grad(): 启用梯度计算
        torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs): 根据前向传播时存储的自动混合精度设置启用自动混合精度
        """
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            """
            x.view_as(x): 创建每个张量的浅拷贝
            """
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 使用浅拷贝调用 run_function 得到 output_tensors
            output_tensors = ctx.run_function(*shallow_copies)
        """
        torch.autograd.grad: 计算梯度
            output_tensors: 作为要计算梯度的函数的输出
            ctx.input_tensors + ctx.input_params: 作为输入，计算梯度的依据
            output_grads: 作为 output_tensors 的梯度
            allow_unused=True: 允许一些输入的梯度可以为 None，如果它们在计算中未被使用
        """
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )

        """
        del: 用于删除不再需要的变量，以释放内存
        """
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors

        return (None, None) + input_grads


def checkpoint(func, x, params, flag):
    """
    梯度检查点
    :param func: forward()方法
    :param x: 传递给forward()方法的参数序列
    :param params: forward()方法的参数
    :param flag: 是否使用梯度检查点
    :return:
    """
    """
    梯度检查点: 一种深度学习优化技术，旨在减少神经网络训练过程中的内存占用
        在训练深度学习模型时，我们需要存储每一层的激活值（即网络层的输出），这样在反向传播时才能计算梯度
        但是，如果网络层数非常多，这些激活值会占用大量的内存
        梯度检查点技术通过只在前向传播时保存部分激活值的信息，而在反向传播时重新计算其他激活值，从而减少了内存的使用
        具体来说，它在前向传播时使用 torch.no_grad() 来告诉PyTorch不需要计算梯度，因为这些激活值会在反向传播时重新计算
        example: 假设你在做一道复杂的数学题，通常你需要写下每一步的计算结果，以便在检查错误时可以追溯回去
                 但如果你确信大部分计算都是正确的，只是在最后几步可能出错，那么你就可以只保存最后几步的结果，然后在检查时重新计算前面的步骤
                 这样，你就可以节省纸张（在神经网络中就是内存）
    """
    if flag:
        args = tuple(x) + tuple(params)
        """
        CheckpointFunction.apply(): apply 方法的参数会被传递给 forward 方法，并在需要时用于反向传播
        """
        return CheckpointFunction.apply(func, len(x), *args)
    else:
        return func(*x)


def conv_nd(dims, *args, **kwargs):
    """
    卷积神经网络
    :param dims: 维数
    :param args: 可变参数
    :param kwargs: 关键字参数
    :return:
    """
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    池化
    :param dims: 维数
    :param args: 可变参数
    :param kwargs: 关键字参数
    :return:
    """
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def group_norm(channels):
    """
    组归一化
    :param channels: 通道数
    :return:
    """
    return GroupNorm(num_groups=32, num_channels=channels)


class GroupNorm(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """

    :param module: 神经网络模型
    :return:
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def time_step_embedding(time_steps, dim, max_period=10000, repeat_only=False):
    """
    时间步嵌入
    :param time_steps: 时间步
    :param dim: 输出向量的维数
    :param max_period: 时间步的最大值
    :param repeat_only:
    :return:
    """
    if not repeat_only:
        half = dim // 2

        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        """
        time_step_embedding[:, None]: 在 time_step_embedding 的最后一个维度上添加一个新的轴
        frequencies[None]: 在 frequencies 的第一个维度上添加一个新的轴
        """
        args = time_step_embedding[:, None].float() * frequencies[None]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            # 如果dim可以被2整除，则说明在计算time_step的embedding时漏掉了一列
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            """
            einops.repeat(tensor, pattern, **axes_lengths): 用于对张量的某个维度进行复制
                tensor: 输入的张量，可以是 NumPy 数组、PyTorch 张量或 TensorFlow 张量
                pattern: 描述如何重复张量的维度
                axes_lengths: 关键字参数，指定每个新维度的长度 
            """
            embedding = einops.repeat(time_steps, 'b -> b d', d=dim)

        return embedding


def count_params(model, verbose=False):
    """
    计算模型的参数量
    :param model: 模型
    :param verbose: 是否打印信息
    :return:
    """
    """
    p.numel(): 返回张量p中的元素数量
        对于一个形状为(a, b, c, d)的张量，它的元素数量为a * b * c * d
    """
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")

    return total_params
