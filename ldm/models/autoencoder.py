# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/15 19:11
# @Software : PyCharm

import pytorch_lightning
import torch
import torch.nn as nn
from ..modules import Encoder, Decoder
from ...ldm import instantiate_from_config, LitEma


class AutoencoderKL(pytorch_lightning.LightningModule):
    def __init__(self,
                 dd_config,
                 loss_config,
                 embedding_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_log_var=False
                 ):
        """
        VQVAE，通常用于图像的编码和解码
        :param dd_config: 编码器和解码器的配置字典，包含模型的结构参数，如通道数、层数等
        :param loss_config: 损失函数的配置字典
        :param embedding_dim: 嵌入维度，即潜在空间的通道数
        :param ckpt_path: 预训练模型的检查点路径，用于加载预训练权重
        :param ignore_keys: 在加载预训练模型时忽略的键列表
        :param image_key: 输入图像的键名
        :param colorize_nlabels: 用于颜色化的标签数量，如果为 None，则不进行颜色化
        :param monitor: 监控指标的名称，用于在训练过程中监控特定指标
        :param ema_decay: 指数移动平均（EMA）的衰减率，用于模型权重的平滑更新
        :param learn_log_var: 是否学习对数方差，用于控制损失函数中的方差项
        """
        super(AutoencoderKL, self).__init__()

        self.learn_log_var = learn_log_var
        self.image_key = image_key
        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.loss = instantiate_from_config(loss_config)

        assert dd_config["double_z"], "double_z不能为None"

        # 量化操作
        self.quant_conv = nn.Conv2d(in_channels=2 * dd_config["z_channels"], out_channels=2 * embedding_dim,
                                    kernel_size=1)
        # 将量化得到的特征恢复到潜在空间
        self.post_quant_conv = torch.nn.Conv2d(in_channels=embedding_dim, out_channels=dd_config["z_channels"],
                                               kernel_size=1)

        self.embedding_dim = embedding_dim

        # 图像的颜色化部分
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1., "ema_decay值的范围在[0, 1]之内"
            self.model_ema = LitEma(self, decay=ema_decay)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys: list = None):
        """

        :param path: 模型路径
        :param ignore_keys: 需要忽略的key
        :return:
        """
        stable_diffusion = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(stable_diffusion.keys())
        for key in keys:
            for ignore_key in ignore_keys:
                if key.startswith(ignore_key):
                    print(f"Deleting key {key} from state_dict.")

                    del stable_diffusion[key]

        self.load_state_dict(stable_diffusion, strict=False)
