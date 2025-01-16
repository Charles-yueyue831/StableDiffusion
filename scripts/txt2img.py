# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import argparse
import os
from omegaconf import OmegaConf
from itertools import islice
from contextlib import nullcontext
import torch
from pytorch_lightning import seed_everything
from ..ldm import instantiate_from_config, DDIMSampler

"""
使用相对导入后，当前模块就不能直接运行，会抛出ValueError: Attempted relative import in non-package的错误
对于编译器来说，它无法理解导入语句中的相对关系，这时候就需要为它说明相对关系，也就是用python -m A.B.C的方式代替python A/B/C.py来运行模块
"""


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    """
    从配置文件中加载模型
    :param config: 模型配置
    :param ckpt: 模型检查点文件的路径
    :param device: CPU or CUDA
    :param verbose: 控制是否打印详细信息
    :return: Latent Diffusion Model
    """
    print(f"从{ckpt}加载模型")

    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    stable_diffusion = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    """
    load_state_dict():
        strict=False: 允许加载的状态字典中的键与模型的参数不完全匹配
    """
    # m包含模型中缺少的键
    # u包含状态字典中多余的键
    m, u = model.load_state_dict(stable_diffusion, strict=False)

    if len(m) > 0 and verbose:
        print(f"missing keys:\n{m}")
    if len(u) > 0 and verbose:
        print(f"unexpected keys:\n{u}")

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        # 如何理解condition_stage_model？
        model.condition_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")

    model.eval()

    return model


def parse_args():
    parser = argparse.ArgumentParser()

    """
    nargs="?": 表示该参数是可选的，允许零个或一个参数值
    """
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="宇航员骑三角龙的专业照片",
        help="指定生成图像的文本提示"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        help="指定保存生成图像的目录",
        default="outputs/txt2img"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="指定DDIM的采样步数",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="指定是否使用 PLMS 采样方法",
    )

    parser.add_argument(
        "--dpm",
        action='store_true',
        help="指定是否使用 DPM (2) 采样器",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="启用该参数后，所有样本将使用相同的起始代码",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="eta 为 0.0 时对应确定性采样",
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="指定采样的次数",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="指定生成图像的高度（以像素为单位）",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="指定生成图像的宽度（以像素为单位）",
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=4,
        help="指定潜在通道的数量",
    )

    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="指定下采样因子，通常为 8 或 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="指定对于每个给定的文本提示生成多少个样本，即batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="指定网格中的行数",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="无条件引导尺度，用于计算生成图像的噪声",
    )

    parser.add_argument(
        "--from_file",
        type=str,
        help="从指定文件中读取文本提示，每个提示占一行",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="指定用于构建模型的配置文件路径",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        help="指定模型检查点的路径",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="指定随机种子，用于可重现的采样",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="指定模型评估的精度",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="指定文件中的每个提示重复生成图像的次数",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="cpu or cuda",
        choices=["cpu", "cuda"],
        default="cpu"
    )

    parser.add_argument(
        "--torchscript",
        action='store_true',
        help="是否使用 TorchScript",
    )

    parser.add_argument(
        "--ipex",
        action='store_true',
        help="是否使用英特尔 ®PyTorch 扩展",
    )

    parser.add_argument(
        "--bf16",
        action='store_true',
        help="是否使用 bfloat16 数据类型",
    )

    opt = parser.parse_args()
    return opt


def chunk(it, size):
    """
    将一个可迭代对象（如列表、字符串等）按照指定的大小size进行分块，返回一个迭代器
    :param it: 要进行分块的可迭代对象
    :param size: 每个分块的大小
    :return:
    """
    """
    iter(): 将可迭代对象转换为迭代器，通过迭代器的方式逐个获取元素
    """
    it = iter(it)
    """
    islice(): 从迭代器it中提取指定数量（size）的元素
    iter(lambda,()): 当lambda函数返回的元组为空时（也就是迭代器it中没有足够的元素来组成一个大小为size的元组时），迭代器停止迭代
    """
    return iter(lambda: tuple(islice(it, size)), ())


def main(args):
    """

    :param args: ArgumentParser()对象
    :return:
    """
    seed_everything(args.seed)

    """
    OmegaConf.load(): 加载配置文件
        OmegaConf 支持 YAML、JSON、INI 等多种配置文件格式
    """
    config = OmegaConf.load(args.config)
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")

    # Latent Diffusion Model
    model = load_model_from_config(config, f"{args.ckpt}", device)

    sampler = DDIMSampler(model=model, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    output_directory = args.out_dir

    batch_size = args.n_samples

    n_rows = args.n_rows if args.n_rows > 0 else batch_size

    if not args.from_file:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"从{args.from_file}中读取提示词")

        with open(args.from_file, "r") as file:
            """
            str.splitlines(): 根据换行符（如 \n、\r\n 等）将字符串按行分割成一个列表
            """
            data = file.read().splitlines()
            data = [p for p in data for i in range(args.repeat)]
            data = list(chunk(data, batch_size))

    sample_directory = os.path.join(output_directory, "samples")
    os.makedirs(sample_directory, exist_ok=True)

    # 如何理解？
    sample_count = 0
    base_count = len(os.listdir(sample_directory))
    grid_count = len(os.listdir(output_directory)) - 1

    start_code = None
    if args.fixed_code:
        # start_code.shape = [batch_size, channel, height, width]
        start_code = torch.randn([args.n_samples, args.channel, args.height // args.factor, args.width // args.factor],
                                 device=device)

    """
    torchscript: 将 PyTorch 模型转换为可以在没有 Python 解释器的情况下运行的格式
    ipex: 英特尔公司为 PyTorch 深度学习框架开发的高性能扩展库
    """
    if args.torchscript or args.ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder

        """
        torch.bfloat16: 使用 16 位来表示一个浮点数，其中，1 位用于符号位，8 位用于指数位，7 位用于尾数位
        torch.float16: 使用 16 位来表示一个浮点数，其中，1 位用于符号位，5 位用于指数位，10 位用于尾数位
        
        torch.cpu.amp.autocast(): PyTorch 的一个上下文管理器，用于自动混合精度（AMP）训练
            在 BF16 模式下，它会自动将某些操作转换为 BF16，以提高计算效率
        nullcontext(): Python 的一个上下文管理器，用于创建一个空的上下文管理器
        """
        additional_context = torch.cpu.amp.autocast() if args.bf16 else nullcontext()
        shape = [args.channel, args.height // args.factor, args.width // args.factor]

        if args.bf16 and not args.torchscript and not args.ipex:
            raise ValueError('Bfloat16仅仅被torchscript+ipex支持')
        if args.bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("如果想在CPU上使用bfloat16，请在启用bf16的情况下使用configs/stable-diffusion/intel/")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("如果想在CPU上运行它，请为模型使用configs/stable diffusion/intel/")

        if args.ipex:
            import intel_extension_for_pytorch as ipex

            bf16_dtype = torch.bfloat16 if args.bf16 else None

            """
            torch.channels_last: 在channels_last内存格式中，通道维度被放置在最后
                                 这种格式在某些硬件（如英特尔的 XPU）上可以提高计算效率，因为它更符合硬件的内存访问模式
            ipex.optimize(...): 英特尔 ® 深度学习加速库（Intel® Extension for PyTorch）
                level="O1": 指定优化级别为O1
                    O1是一种相对温和的优化级别，它会进行一些基本的优化操作，以提高模型的运行速度，同时保持模型的准确性
                auto_kernel_selection: 让ipex自动选择适合的内核，以进一步提高性能
            """
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if args.torchscript:
            with torch.no_grad(), additional_context:
                if unet.use_checkpoint:
                    raise ValueError("渐变检查点不适用于跟踪，为模型使用configs/stable-diffusion/intel/或在配置中禁用检查点")

                # image: 作为输入数据用于后续对unet模型的跟踪
                image = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                # time: 表示时间步信息
                time = torch.ones(2, dtype=torch.int64)
                # context: 包含一些上下文信息
                context = torch.ones(2, 77, 1024, dtype=torch.float32)

                """
                torch.jit.trace(): 通过给定的输入示例来生成一个脚本化的模型，脚本化的模型可以提高模型的运行效率，并且便于在不同环境中部署
                """
                scripted_unet = torch.jit.trace(unet, (image, time, context))

                """
                torch.jit.optimize_for_inference(): 对脚本化模型进行一系列的优化，以进一步提高模型在推理阶段的性能，例如优化计算图、减少内存使用
                """
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)

                # 如何理解scripted_diffusion_model？
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, samples_ddim)
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)

                # 如何理解first_stage_model？
                model.first_stage_model.decoder = scripted_decoder





