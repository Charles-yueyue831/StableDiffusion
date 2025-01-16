# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import torch
import numpy as np
from tqdm import tqdm
from ...modules import make_ddim_timesteps, make_ddim_sampling_parameters


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        """
        DDIM采样器
        :param model: Latent Diffusion Model
        :param schedule: 调度方式
        :param device: CPU or CUDA
        :param kwargs: 可变参数
        """
        super().__init__()

        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr, desc=""):
        """

        :param name: 属性名
        :param attr: 属性值
        :param desc: 对属性的描述
        :return:
        """
        if isinstance(attr, torch.Tensor):
            if attr.device != self.device:
                attr = attr.to(self.device)

        """
        setattr(self, name, attr): 将attr设置为实例的属性，属性名为name
        """
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        """
        DDIM 采样的参数
        :param ddim_num_steps: ddim采样步数
        :param ddim_discretize: 离散化方法
        :param ddim_eta: 一个控制采样随机性的参数
        :param verbose: 是否打印详细信息
        :return:
        """
        # make_ddim_timesteps: 根据给定的离散化方法生成 DDIM 采样的时间步
        self.ddim_timesteps = make_ddim_timesteps(ddim_discrete_method=ddim_discretize,
                                                  num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        # \bar{\alpha}: 获取模型的累积乘积alphas_cumprod
        alphas_cumprod = self.model.alphas_cumprod

        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, '必须为每个时间步定义alpha'

        """
        x.clone(): 深拷贝
        x.detach(): 从计算图中分离，不参与梯度的计算
        """
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas), desc="扩散系数")
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod), desc="时刻t的alpha的累积乘积")
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev),
                             desc="时刻t_1的alpha的累积乘积")

        # 正向扩散过程
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())),
                             desc="正向扩散过程中的均值")
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())),
                             desc="正向扩散过程中的标准差")
        # 如何理解log_one_minus_alphas_cumprod？
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())),
                             desc="逆向采样过程中根据x_t计算x_0")
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)),
                             desc="逆向采样过程中根据x_t计算x_0")

        # DDIM的采样参数
        # 选择alphas计算方差
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alpha_cumprod=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)

        self.register_buffer('ddim_sigmas', ddim_sigmas, desc="DDIM采样阶段的方差")
        self.register_buffer('ddim_alphas', ddim_alphas, desc=r"DDIM扩散过程中的\bar{\alpha}_t")
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev, desc=r"DDIM扩散过程中的\bar{\alpha}_{t-1}")
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas),
                             desc=r"DDIM扩散过程中的1-\bar{\alpha}_t")

        # DDIM中的方差
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) * (1 - ddim_alphas / ddim_alphas_prev))

        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps,
                             desc="DDIM中的方差")

    @torch.no_grad()
    def sample(self,
               n_steps,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        """

        :param n_steps: DDIM 采样的步数
        :param batch_size: 每次采样的样本数量
        :param shape: 样本形状(通道数, 高度, 宽度)
        :param conditioning: 条件信息，用于指导采样过程
        :param callback: 回调函数
        :param normals_sequence: 如何理解normals_sequence？
        :param img_callback: 另一个回调函数，专门用于处理生成的图像样本
        :param quantize_x0: 是否对初始样本（x0）进行量化操作
        :param eta: 控制采样随机性的参数，取值范围通常在 0 到 1 之间
        :param mask: 如何理解mask？
        :param x0: 初始样本，如果为None，则会随机生成初始样本
        :param temperature: 用于调整噪声强度，值为 1 时表示正常噪声强度，值越大，噪声对采样结果的影响越大
        :param noise_dropout: 随机丢弃噪声的概率值，值为 0 时表示不丢弃噪声，取值在 0 到 1 之间
        :param score_corrector: 如何理解score_corrector？
        :param corrector_kwargs: 包含传递给score_corrector的参数
        :param verbose: 是否打印详细的采样信息
        :param x_T: 在采样开始时的初始状态，通常是一个噪声张量
        :param log_every_t: 每log_every_t步打印一次采样信息
        :param unconditional_guidance_scale: 无条件引导尺度，用于控制无条件引导的强度，值为 1 时表示不进行额外的无条件引导
        :param unconditional_conditioning: 如何理解unconditional_conditioning？
        :param dynamic_threshold: 如何理解dynamic_threshold？
        :param ucg_schedule: 如何理解ucg_schedule？
        :param kwargs:
        :return:
        """
        if conditioning is not None:
            # 检查conditioning是否是字典类型
            if isinstance(conditioning, dict):
                # 从字典conditioning中获取第一个键对应的值，并赋值给tmp
                tmp = conditioning[list(conditioning.keys())[0]]
                # 如果tmp是列表类型，则不断获取列表中的第一个元素，直到tmp不再是列表
                while isinstance(tmp, list):
                    tmp = tmp[0]

                if tmp.shape[0] != batch_size:
                    print(f"Warning: Got {tmp.shape[0]} conditions but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for tmp in conditioning:
                    if tmp.shape[0] != batch_size:
                        print(f"Warning: Got {tmp.shape[0]} conditions but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditions but batch-size is {batch_size}")

            # DDIM 采样的参数
            self.make_schedule(ddim_num_steps=n_steps, ddim_eta=eta, verbose=verbose)

            # sampling
            channel, height, width = shape
            size = (batch_size, channel, height, width)
            print(f'DDIM采样的数据类型: {size}\teta {eta}')

    @torch.no_grad()
    def ddim_sampling(self,
                      condition,
                      shape,
                      x_T=None,
                      ddim_use_original_steps=False,
                      callback=None,
                      timesteps=None,
                      quantize_denoised=False,
                      mask=None,
                      x0=None,
                      img_callback=None,
                      log_every_t=100,
                      temperature=1.,
                      noise_dropout=0.,
                      score_corrector=None,
                      corrector_kwargs=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      dynamic_threshold=None,
                      ucg_schedule=None):
        """

        :param condition: 条件信息，用于指导采样过程，例如文本描述对应的特征向量等
        :param shape: 生成样本的形状(batch_size, channels, height, width)
        :param x_T: 初始噪声样本
        :param ddim_use_original_steps: 是否使用原始的 DDPM 时间步，还是使用 DDIM 自定义的时间步
        :param callback: 回调函数
        :param timesteps: 采样的时间步序列
        :param quantize_denoised: 是否对去噪后的样本进行量化操作
        :param mask: 如何理解mask？
        :param x0: 初始样本
        :param img_callback: 另一个回调函数，专门用于处理生成的图像样本
        :param log_every_t: 每log_every_t步打印一次采样信息
        :param temperature: 调整噪声强度的参数，值为 1 表示正常噪声强度，值越大，噪声对采样结果的影响越大
        :param noise_dropout: 用于随机丢弃噪声，取值在 0 到 1 之间，默认为 0，表示不丢弃噪声
        :param score_corrector: 如何理解score_corrector？
        :param corrector_kwargs: 一个字典，包含传递给score_corrector的参数
        :param unconditional_guidance_scale: 无条件引导尺度，用于控制无条件引导的强度，值为 1 表示不进行额外的无条件引导
        :param unconditional_conditioning: 如何理解unconditional_conditioning？
        :param dynamic_threshold: 如何理解dynamic_threshold？
        :param ucg_schedule: 如何理解ucg_schedule？
        :return:
        """
        # self.model: Latent Diffusion Model
        device = self.model.betas.device

        batch_size = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            # DDPM的采样步数 or DDIM的采样步数
            n_steps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            # self.ddim_timesteps.shape[0]表示DDIM的采样步数
            # 如何理解subset_end？
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            n_steps = self.ddim_timesteps[:subset_end]
