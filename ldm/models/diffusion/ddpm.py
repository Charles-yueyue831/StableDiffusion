# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import itertools
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning
from ....ldm import instantiate_from_config, count_params, make_beta_schedule, default
from ...modules import LitEma


def disabled_train(self):
    """
    模型的训练/评估模式将保持不变，不会像正常的 train() 或 eval() 方法那样启用训练模式或评估模式
    :param self:
    :return:
    """
    return self


class DDPM(pytorch_lightning.LightningModule):
    def __init__(self,
                 unet_config,
                 time_steps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 condition_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encoding=False,
                 learn_log_var=False,
                 log_var_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 ):
        """

        :param unet_config: UNet模型
        :param time_steps: 扩散过程中的时间步数
        :param beta_schedule: beta调度器
        :param loss_type: 模型训练过程中使用的损失函数类型
        :param ckpt_path: 模型的检查点路径
        :param ignore_keys: 在加载模型检查点时需要忽略的键名列表
        :param load_only_unet: 是否只加载 U-Net 模型的权重
        :param monitor: 在训练过程中用于监控的指标
        :param use_ema: 是否使用指数移动平均（EMA）技术
        :param first_stage_key: 获取第一阶段数据（通常是图像数据）的键名
        :param image_size: 输入图像的大小
        :param channels: 输入图像的通道数
        :param log_every_t: 记录日志的时间间隔
        :param clip_denoised: 是否对去噪后的图像进行裁剪操作
        :param linear_start: 在beta值的调度中，指定beta的起始值
        :param linear_end: 在beta值的调度中，指定beta的结束值
        :param cosine_s: 在beta值的余弦调度中，指定一个控制参数
        :param given_betas: 如果提供了预定义的beta值序列，则使用该序列作为扩散过程中的beta值
        :param original_elbo_weight: 如何理解original_elbo_weight？原始证据下限（ELBO）损失的权重。在计算总损失时，ELBO 损失会乘以这个权重
        :param v_posterior: 控制后验方差的调整比例
        :param l_simple_weight: 如何理解l_simple_weight？简单损失的权重，在计算总损失时，简单损失会乘以这个权重
        :param condition_key: 条件信息的键名
        :param parameterization: 模型的预测模式，目前支持"eps"、"x0"和"v"三种模式
        :param scheduler_config: 学习率调度器的配置
        :param use_positional_encoding: 是否使用位置编码
        :param learn_log_var: 如何理解learn_log_var？是否学习对数方差
        :param log_var_init: 如何理解log_var_init？对数方差的初始值
        :param make_it_fit: 若为True则尝试使加载的参数形状适配当前模型
        :param ucg_training: 无条件引导训练的配置
        :param reset_ema: 是否重置指数移动平均（EMA）的状态
        :param reset_num_ema_updates: 是否重置 EMA 的更新次数
        """
        super(DDPM, self).__init__()

        """
        eps: 预测噪声
        x0: 预测原始数据
        
        """
        assert parameterization in ["eps", "x0", "v"], '目前仅支持"eps"、"x0"和"v"'
        self.parameterization = parameterization

        print(f"{self.__class__.__name__}: 在{self.parameterization}预测模式下运行")

        self.condition_stage_model = None

        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encoding = use_positional_encoding

        self.model = DiffusionWrapper(unet_config, condition_key)

        # 计算UNet模型的参数量
        count_params(self.model, verbose=True)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        self.make_it_fit = make_it_fit

        if reset_ema:
            assert ckpt_path is not None, "ckpt的路径不能为None"

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

            if reset_ema:
                assert self.use_ema, "用户没有选择使用EMA"
                print("将ema重置为纯模型权重")
                self.model_ema = LitEma(self.model)

        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema, "用户没有选择使用EMA"
            self.model_ema.reset_num_updates()

        self.num_time_steps = time_steps
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, time_steps=time_steps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        # L1正则化 or L2正则化
        self.loss_type = loss_type

        self.learn_log_var = learn_log_var
        """
        torch.full(size, fill_value): 创建一个填充了特定值的张量
            size: 指定创建的张量的形状
            fill_value: 用于填充张量的每个元素
        """
        self.log_var = torch.full(fill_value=log_var_init, size=(self.num_timesteps,))

        if self.learn_log_var:
            self.log_var = nn.Parameter(self.log_var, requires_grad=True)

        # ucg_training: 无条件引导训练的配置
        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            """
            np.random.RandomState(): 创建一个 RandomState 对象，这个对象可以生成随机数
                可以通过传递一个种子（seed）来初始化这个对象，确保每次生成的随机数序列是相同的
            """
            self.ucg_prng = np.random.RandomState()

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        """

        :param path: ckpt文件的路径
        :param ignore_keys: 需要忽略的key
        :param only_model: 是否只加载UNet模型
        :return:
        """
        if ignore_keys is None:
            ignore_keys = []

        stable_diffusion = torch.load(path, map_location="cpu")

        if "state_dict" in list(stable_diffusion.keys()):
            stable_diffusion = stable_diffusion["state_dict"]

        keys = list(stable_diffusion.keys())
        for key in keys:
            for ignore_key in ignore_keys:
                if key.startswith(ignore_key):
                    print(f"从state_dict中删除{key}")
                    del stable_diffusion[key]

        # 如果不理解这段代码也不影响对模型整体的理解
        # 检查make_it_fit属性是否为True，若为True则尝试使加载的参数形状适配当前模型
        if self.make_it_fit:
            """
            self.named_parameters(): 返回一个生成器，包含参数的名称和对应的张量
            self.named_buffers(): 返回一个生成器，包含缓冲区的名称和对应的张量
            itertools.chain: 接受多个迭代器作为参数，返回一个新的迭代器，该迭代器依次生成所有输入迭代器中的元素
            """
            n_params = len([name for name, _ in itertools.chain(self.named_parameters(), self.named_buffers())])

            for name, param in tqdm(itertools.chain(self.named_parameters(), self.named_buffers()),
                                    desc="将旧weights与新weights进行匹配",
                                    total=n_params):
                if name not in stable_diffusion:
                    continue

                """
                形状匹配:
                    确保旧参数和新参数的维度数相同
                    如果新参数的维度数大于2，确保从第3维开始的形状相同
                形状不匹配:
                    克隆新参数
                    根据旧参数的形状，将旧参数的值复制到新参数中
                    如果新参数是1维的，直接复制
                    如果新参数是2维或更高维的，进行双重循环复制
                    计算每个旧参数值在新参数中使用的次数
                    计算每个新参数值在旧参数中使用的次数
                    将新参数除以使用次数，进行归一化
                """

                old_shape = stable_diffusion[name].shape
                new_shape = param.shape

                assert len(old_shape) == len(new_shape), "确保旧参数和新参数的维度数相同"

                if len(new_shape) > 2:
                    assert new_shape[2:] == old_shape[2:], "如果新参数的维度数大于2，确保从第3维开始的形状相同"

                if not new_shape == old_shape:
                    # 如果新参数和旧参数的形状不匹配
                    new_param = param.clone()
                    old_param = stable_diffusion[name]

                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            # 遍历新参数的每个元素
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    stable_diffusion[name] = new_param

        """
        missing: 包含在模型中存在但在预训练权重中不存在的键
        unexpected: 包含在预训练权重中存在但在模型中不存在的键
        """
        missing, unexpected = self.load_state_dict(stable_diffusion, strict=False) if not only_model \
            else self.model.load_state_dict(stable_diffusion, strict=False)

        if len(missing) > 0:
            print(f"在模型中存在但在预训练权重中不存在的键:\n {missing}")
        if len(unexpected) > 0:
            print(f"在预训练权重中存在但在模型中不存在的键:\n {unexpected}")

    def register_schedule(self, given_betas=None, beta_schedule="linear", time_steps=1000, linear_start=1e-4,
                          linear_end=2e-2, cosine_s=8e-3):
        """

        :param given_betas: 如果提供了预定义的beta值序列，则使用该序列作为扩散过程中的beta值
        :param beta_schedule: beta调度器
        :param time_steps: 扩散过程中的时间步数
        :param linear_start: 在beta值的调度中，指定beta的起始值
        :param linear_end: 在beta值的调度中，指定beta的结束值
        :param cosine_s: 在beta值的余弦调度中，指定一个控制参数
        :return:
        """
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, time_steps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)

        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.append(1., alphas_bar[:-1])

        n_steps, = betas.shape
        self.num_time_steps = n_steps

        assert alphas_bar.shape[0] == self.num_time_steps, '必须为每个时间步定义alpha'

        """
        partial(func,  *args, **kwargs): 创建一个预设参数的函数
            func: 需要设置参数的函数
            *args, **kwargs: 预设的参数
        """
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer_desc(name='betas', tensor=to_torch(betas), desc="时间步t的扩散系数")
        self.register_buffer_desc(name='alphas_bar', tensor=to_torch(alphas_bar), desc="时间步t的alpha的累积乘积")
        self.register_buffer_desc(name='alphas_bar_prev', tensor=to_torch(alphas_bar_prev),
                                  desc="时间步t-1的alpha的累积乘积")

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer_desc(name='sqrt_alphas_bar', tensor=to_torch(np.sqrt(alphas_bar)),
                                  desc="正向扩散过程中的均值")
        self.register_buffer_desc(name='sqrt_one_minus_alphas_bar', tensor=to_torch(np.sqrt(1. - alphas_bar)),
                                  desc="正向扩散过程中的标准差")
        self.register_buffer_desc(name='log_one_minus_alphas_bar', tensor=to_torch(np.log(1. - alphas_bar)))
        self.register_buffer_desc(name='sqrt_recip_alphas_bar', tensor=to_torch(np.sqrt(1. / alphas_bar)),
                                  desc="逆向采样过程中根据x_t计算x_0")
        self.register_buffer_desc(name='sqrt_recipm1_v', tensor=to_torch(np.sqrt(1. / alphas_bar - 1)),
                                  desc="逆向采样过程中根据x_t计算x_0")

        # 计算后验分布q(x_{t-1} | x_t, x_0)
        """
        self.v_posterior: 控制后验方差的调整比例
            噪声去除过强（self.v_posterior 大）: 当噪声去除过强时，模型可能会误去除一些有价值的信息，导致生成的结果模糊或失真
            噪声去除过弱（self.v_posterior 小）: 当噪声去除过弱时，模型可能会留下一些不必要的噪声，这可能导致最终的生成结果含有不必要的杂散噪声
        """
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_bar_prev) / (
                1. - alphas_bar) + self.v_posterior * betas
        self.register_buffer_desc(name='posterior_variance', tensor=to_torch(posterior_variance),
                                  desc="逆向采样过程中的后验方差")

        """
        posterior_log_variance_clipped: 直接计算得到的后验方差可能在数值上不稳定，导致训练过程中的梯度爆炸或消失等问题
            为此，DDPM 对计算得到的后验方差进行裁剪，确保其在合理范围内，以提高数值稳定性
        """
        self.register_buffer_desc(name='posterior_log_variance_clipped',
                                  tensor=to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        self.register_buffer_desc(name='posterior_mean_coef1',
                                  tensor=to_torch(betas * np.sqrt(alphas_bar_prev) / (1. - alphas_bar)),
                                  desc="逆向采样均值式子中x_0前的系数")
        self.register_buffer_desc(name='posterior_mean_coef2',
                                  tensor=to_torch((1. - alphas_bar_prev) * np.sqrt(alphas) / (1. - alphas_bar)),
                                  desc="逆向采样均值式子中x_t前的系数")

        # 可以不用考虑这部分的内容
        if self.parameterization == "eps":
            # 预测噪声
            # lvlb_weights: 在DDPM论文中。模型的损失函数前的系数
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - alphas_bar))

        elif self.parameterization == "x0":
            # 预测原始图片
            lvlb_weights = 0.5 * np.sqrt(to_torch(alphas_bar)) / (2. * 1 - to_torch(alphas_bar))

        elif self.parameterization == "v":
            # v-prediction
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - alphas_bar)))

        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        """
        self.register_buffer(name, tensor, persistent)
            persistent=False: 表示这个缓冲区不会在模型的state_dict中持久化
        """
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        """
        torch.isnan(tensor): 返回一个与tensor形状相同的布尔张量，其中每个元素表示tensor中对应位置的元素是否为NaN（Not a Number）
        .all(): 检查这个布尔张量中的所有元素是否都为True
        """
        assert not torch.isnan(self.lvlb_weights).all(), "lvlb_weights中存在NaN"

    def register_buffer_desc(self, name, tensor, desc=""):
        """

        :param name: 缓冲区参数的名称
        :param tensor: 参数名称对应的tensor
        :param desc: 对该缓冲区参数的描述
        :return:
        """
        self.register_buffer(name, tensor)


class DiffusionWrapper(pytorch_lightning.LightningModule):
    def __init__(self, diffusion_model_config, condition_key):
        """
        不同的条件对应不同的数据处理方式
        :param diffusion_model_config: UNet模型的配置
        :param condition_key: 条件信息的键名
        """
        super(DiffusionWrapper, self).__init__()

        self.sequential_cross_attn = diffusion_model_config.pop("sequential_cross_attn", False)
        # UNet模型
        self.diffusion_model = instantiate_from_config(diffusion_model_config)
        self.condition_key = condition_key

        assert self.condition_key in [None, 'concat', 'cross_attn', 'hybrid', 'adm', 'hybrid-adm', 'cross_attn-adm']

    def forward(self, x, t, condition_concat: list = None, condition_cross_attn: list = None, condition_adm=None):
        """

        :param x: 输入的噪声数据 query向量
        :param t: 时间步
        :param condition_concat: 需要与x进行拼接的图像特征
        :param condition_cross_attn: 条件数据 key向量和value向量
        :param condition_adm: 标签数据
        :return:
        """

        # 如果条件数据为空
        if self.condition_key is None:
            out = self.diffusion_model(x, t)
        elif self.condition_key == "concat":
            """
            将噪声数据和条件数据通过torch.cat进行拼接
            
            假设 x 是一个形状为 (2, 3) 的张量，c_concat 是一个包含两个形状为 (2, 2) 的张量的列表
                [x] + c_concat 将生成一个包含三个张量的列表，第一个张量是 x，后续两个张量是 c_concat 中的张量
            """
            x_condition = torch.cat([x] + condition_concat, dim=1)
            out = self.diffusion_model(x_condition, t)

        elif self.condition_key == 'cross_attn':
            if not self.sequential_cross_attn:
                """
                将不同的条件数据通过torch.cat进行拼接
                """
                cross_attn_condition = torch.cat(condition_cross_attn, dim=1)
            else:
                cross_attn_condition = condition_cross_attn

            out = self.diffusion_model(x, t, context=cross_attn_condition)

        elif self.condition_key == 'hybrid':
            """
            将噪声数据和条件数据通过torch.cat进行拼接
            将不同的条件数据通过torch.cat进行拼接
            """
            x_condition = torch.cat([x] + condition_concat, dim=1)

            if not self.sequential_cross_attn:
                cross_attn_condition = torch.cat(condition_cross_attn, dim=1)
            else:
                cross_attn_condition = condition_cross_attn

            out = self.diffusion_model(x_condition, t, context=cross_attn_condition)

        elif self.condition_key == 'hybrid-adm':
            """
            外条件（Additional Conditioning）: 在生成图像时，除了文本提示之外，提供给模型的其他形式的输入
                这些输入可以是图像、姿态图、深度图、草图等
            """
            # hybrid-adm: 结合堆叠、交叉注意力和额外条件信息
            assert condition_adm is not None, "结合堆叠、交叉注意力和额外条件信息"
            x_condition = torch.cat([x] + condition_concat, dim=1)

            if not self.sequential_cross_attn:
                cross_attn_condition = torch.cat(condition_cross_attn, dim=1)
            else:
                cross_attn_condition = condition_cross_attn

            out = self.diffusion_model(x_condition, t, context=cross_attn_condition, label=condition_adm)

        elif self.conditioning_key == 'cross_attn-adm':
            # cross_attn-adm: 结合交叉注意力和额外条件信息
            assert condition_adm is not None, "结合交叉注意力和额外条件信息"

            if not self.sequential_cross_attn:
                cross_attn_condition = torch.cat(condition_cross_attn, dim=1)
            else:
                cross_attn_condition = condition_cross_attn

            out = self.diffusion_model(x, t, context=cross_attn_condition, label=condition_adm)

        elif self.conditioning_key == 'adm':
            # adm: 仅使用额外条件信息
            cross_attn_condition = condition_cross_attn[0]
            out = self.diffusion_model(x, t, label=cross_attn_condition)

        else:
            raise NotImplementedError()

        return out


class LatentDiffusion(DDPM):
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 condition_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        """

        :param first_stage_config: 第一阶段模型的配置信息，用于实例化第一阶段模型
        :param cond_stage_config: 条件阶段模型的配置信息，决定条件阶段模型的构建方式
        :param num_timesteps_cond: 条件时间步数量
        :param cond_stage_key: 条件阶段数据的键
        :param cond_stage_trainable: 条件阶段模型是否可训练
        :param concat_mode: 连接模式标志
        :param cond_stage_forward: 条件阶段模型前向传播
        :param condition_key: [None, 'concat', 'cross_attn', 'hybrid', 'adm', 'hybrid-adm', 'cross_attn-adm']
        :param scale_factor: 缩放因子
        :param scale_by_std: 是否按标准差缩放
        :param force_null_conditioning: 是否强制使用空条件
        :param args:
        :param kwargs:
        """

        self.first_stage_model = None
        self.cond_stage_model = None

        self.force_null_conditioning = force_null_conditioning
        self.num_time_steps_cond = default(num_timesteps_cond, 1)

        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps'], "条件时间步的数量必须小于等于时间步的数量"

        if condition_key is None:
            # concat_mode: 条件数据和图像数据拼接的模式
            condition_key = 'concat' if concat_mode else 'cross_attn'
        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            # 如果cond_stage_config为'__is_unconditional__'并且不是强制使用空条件
            condition_key = None

        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        # ignore_keys: 可以被state_dict忽略的key
        ignore_keys = kwargs.pop("ignore_keys", [])

        super(LatentDiffusion, self).__init__(condition_key=condition_key, *args, **kwargs)

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0

        if not scale_by_std:
            # 如果不是按照标准差进行缩放
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        # cond_stage_forward: 条件阶段模型的前向传播方法
        self.cond_stage_forward = cond_stage_forward
        # clip_denoised: 控制在去噪过程中是否对生成的图像进行裁剪
        self.clip_denoised = False
        # 用于处理边界框信息的tokenizer
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema, "使用指数移动平均"
                self.model_ema = LitEma(self.model)

        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema, "使用指数移动平均"
            self.model_ema.reset_num_updates()

    def instantiate_first_stage(self, config):
        """
        实例化第一阶段的模型，通常是一个编码器或生成器，用于将输入数据编码或生成为潜在表示
        :param config: 配置文件
        :return:
        """
        model = instantiate_from_config(config)

        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        """
        实例化条件阶段的模型，通常是一个编码器，用于将条件信息编码为潜在表示
        :param config: 配置文件
        :return:
        """
        # cond_stage_trainable: 条件阶段模型是否可训练
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("使用第一阶段的模型作为条件阶段的模型")
                self.cond_stage_model = self.first_stage_model

            elif config == "__is_unconditional__":
                print(f"作为一个无条件模型训练{self.__class__.__name__}")
                self.cond_stage_model = None

            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False

        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def forward(self, x, condition, *args, **kwargs):
        """

        :param x: 输入数据
        :param condition: 条件数据
        :param args:
        :param kwargs:
        :return:
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        if self.model.condition_key is not None:
            assert condition is not None,"条件数据不能为None"

        if self.cond_stage_trainable:
            condition = self.get_learned_conditioning(condition)

    def get_learned_conditioning(self, condition):
        if self.cond_stage_forward is None:
            """
            callable(): 检查一个对象是否可以被调用
                如果一个对象定义了__call__方法，那么这个对象就是可调用的
            """
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                condition = self.cond_stage_model.encode(condition)
                if isinstance(condition, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
