# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/13 13:35
# @Software : PyCharm

import math
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import SpatialTransformer
from ....ldm import conv_nd, group_norm, avg_pool_nd, checkpoint, zero_module, time_step_embedding

"""
多头注意力机制中每一个注意力机制关注不同的channel
"""


class TimestepBlock(nn.Module):
    """
    将时间步嵌入作为forward()第二个参数的任何模块
    """

    @abstractmethod
    def forward(self, x, embedding):
        """

        :param x: query向量
        :param embedding: 时间步嵌入信息
        :return:
        """
        """
        抽象方法，规定继承TimestepBlock的类应该包含的方法
        """


class TimestepEmbeddingSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将时间步嵌入作为额外输入传递给支持它的子模块
    """
    """
    nn.Sequential 是 PyTorch 中的一个容器，它按顺序存储子模块，在调用 forward 方法时会依次调用这些子模块的 forward 方法
    """

    def forward(self, x, embedding, context=None):
        """

        :param x: query向量
        :param embedding: 时间步嵌入
        :param context: key向量和value向量
        :return:
        """
        """
        signature 通常指的是函数或方法的参数列表
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, embedding)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        """
        上采样
        :param channels: 输入通道数或输出通道数
        :param use_conv: 是否使用卷积神经网络
        :param dims: 卷积神经网络的维数
        :param out_channels: 输出通道数
        :param padding: 卷积神经网络的填充
        """
        super(UpSample, self).__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, kernel_size=3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels, "输入数据通道的维数必须和输入通道数相同"

        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )

        else:
            """
            F.interpolate(): 对张量进行上采样或下采样
                scale_factor: 指定每个空间维度的放大倍数
                mode: 支持多种采样模式，如最近邻插值（nearest）、双线性插值（bilinear）、双三次插值（bicubic）
                
                example:
                    # 创建一个输入张量
                    x = torch.randn(1, 3, 8, 8)
                    
                    # 使用 F.interpolate 进行上采样
                    x_upsampled = F.interpolate(x, scale_factor=2, mode="nearest")
                    
                    # 打印输出张量的形状
                    print(x_upsampled.shape)  # 输出：torch.Size([1, 3, 16, 16])
            """
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        """
        下采样
        :param channels: 输入通道数或输出通道数
        :param use_conv: 是否使用卷积神经网络
        :param dims: 卷积神经网络的维数
        :param out_channels: 输出通道数
        :param padding: 卷积神经网络的填充
        """
        super(DownSample, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        # 三维卷积神经网络 (深度、高度、宽度)
        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.operation = conv_nd(
                dims, self.channels, self.out_channels, kernel_size=3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels, "输入通道数必须与输出通道数相同"
            self.operation = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels, "输入数据通道的维数必须和输入通道数相同"
        return self.operation(x)


class ResBlock(TimestepBlock):
    def __init__(
            self,
            channels,
            embedding_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        """
        残差连接
        :param channels: 输入通道数
        :param embedding_channels: 时间步通道数
        :param dropout: 随机Dropout概率
        :param out_channels: 输出通道数
        :param use_conv: 在skip connection中使用spatial convolution而不是1x1 convolution
        :param use_scale_shift_norm: 是否使用FiLM机制
        :param dims: 卷积神经网络的维数
        :param use_checkpoint: 是否使用检查点
        :param up: 上采样
        :param down: 下采样
        """
        super(ResBlock, self).__init__()
        self.channels = channels
        self.embedding_channels = embedding_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        """
        组归一化
        Swish激活函数
        卷积神经网络
        """
        self.in_layers = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size=3, padding=1),
        )

        self.updown = up or down

        """
        上采样和下采样的过程中如果使用卷积神经网络，则通道数从channels变成out_channels
        """
        if up:
            self.h_up_down = UpSample(channels=channels, use_conv=False, dims=dims)
            self.x_up_down = UpSample(channels=channels, use_conv=False, dims=dims)
        elif down:
            self.h_up_down = DownSample(channels=channels, use_conv=False, dims=dims)
            self.x_up_down = DownSample(channels=channels, use_conv=False, dims=dims)
        else:
            self.h_up_down = self.x_up_down = nn.Identity()

        """
        组归一化
        Swish激活函数
        """
        self.embedding_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                in_features=embedding_channels,
                out_features=2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        """
        组归一化
        Swish激活函数
        将模型的参数归零
        """
        self.out_layers = nn.Sequential(
            group_norm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size=3, padding=1)
            ),
        )

        """
        如果out_channels和channels相同，则恒等映射；如果不同，则使用卷积神经网络将channels映射到out_channels
        """
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size=3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size=1)

    def forward(self, x, embedding):
        """

        :param x: 输入数据
        :param embedding: 时间步嵌入
        :return:
        """
        """
        梯度检查点可以节省内存
        """
        return checkpoint(self._forward, (x, embedding), self.parameters(), self.use_checkpoint)

    def _forward(self, x, embedding):
        """

        :param x: 输入数据
        :param embedding: 时间步嵌入
        :return:
        """
        if self.updown:
            # in_rest: 组归一化 + Swish激活函数; in_conv: 卷积神经网络
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_up_down(h)
            x = self.x_up_down(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # embedding_layers: 组归一化 + Swish激活函数
        embedding_out = self.embedding_layers(embedding).type(h.dtype)

        while len(embedding_out.shape) < len(h.shape):
            """
            ...: 表示前面所有的维度
            """
            embedding_out = embedding_out[..., None]

        # FiLM机制
        if self.use_scale_shift_norm:
            # out_norm: 组归一化; out_rest: Swish激活函数 + 将模型的参数归零
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]

            scale, shift = torch.chunk(embedding_out, 2, dim=1)

            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = h + embedding_out
            # out_layers(): 组归一化 + Swish激活函数 + 将模型的参数归零
            h = self.out_layers(h)

        # 残差连接
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        """

        :param n_heads: 注意力头的数量
        """
        super(QKVAttention, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """

        :param qkv: [N x (3 * H * C) x T]
        :return:
        """
        batch_size, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0, "qkv.shape[1]的维数不能整除3*注意力头的数量"

        # q、k、v的通道数
        channel = width // (3 * self.n_heads)

        q, k, v = qkv.chunk(3, dim=1)

        scale = 1 / math.sqrt(length)
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(batch_size * self.n_heads, channel, length),
            (k * scale).view(batch_size * self.n_heads, channel, length),
        )

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        result = torch.einsum("bts, bcs->bct", weight, v.reshape(batch_size * self.n_heads, channel, length))

        return result.reshape(batch_size, -1, length)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        """

        :param n_heads: 注意力头的数量
        """
        super(QKVAttentionLegacy, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """

        :param qkv: [N x (3 * H * C) x T]
        :return:
        """
        batch_size, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0, "qkv.shape[1]的维数不能整除3*注意力头的数量"

        # q、k、v的通道数
        channel = width // (3 * self.n_heads)

        # 先分割注意力头再分开QKV向量
        q, k, v = qkv.reshape(batch_size * self.n_heads, channel * 3, length).split(channel, dim=1)

        scale = 1 / math.sqrt(length)

        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        result = torch.einsum("bts,bcs->bct", weight, v)

        return result.reshape(batch_size, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        """
        注意力块
        :param channels: 输入通道数
        :param num_heads: 注意力头的数量
        :param num_head_channels: 注意力头的通道数
        :param use_checkpoint: 是否使用梯度检查点
        :param use_new_attention_order:
        """
        super(AttentionBlock, self).__init__()

        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads

        else:
            assert channels % num_head_channels == 0, f"q,k,v的通道数{channels}不能被注意力头的通道数{num_head_channels}整除"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = group_norm(channels=channels)
        self.qkv = conv_nd(1, channels, channels * 3, kernel_size=1)

        if use_new_attention_order:
            # 在分割注意力头之前分开QKV向量
            self.attention = QKVAttention(self.num_heads)
        else:
            # 在分开QKV向量之前分割注意力头
            self.attention = QKVAttentionLegacy(self.num_heads)

        """
        一维卷积: 用于处理序列数据，如时间序列、音频信号或文本数据
        example:
            假设有一个输入序列 [1, 2, 3, 4, 5]，卷积核 [1, 0, -1]，卷积核的大小为 3
            
            第一次滑动: 1*1 + 2*0 + 3*(-1) = 1 - 3 = -2
            第二次滑动: 2*1 + 3*0 + 4*(-1) = 2 - 4 = -2
            第三次滑动: 3*1 + 4*0 + 5*(-1) = 3 - 5 = -2
        """
        self.proj_out = zero_module(conv_nd(1, channels, channels, kernel_size=1))

    def forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        # spatial: height + width
        batch_size, channel, *spatial = x.shape

        x = x.reshape(batch_size, channel, -1)

        qkv = self.qkv(self.norm(x))

        h = self.attention(qkv)
        h = self.proj_out(h)

        return (x + h).reshape(batch_size, channel, *spatial)


class UNetModel(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            use_bf16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_up_sample=-1,
            use_scale_shift_norm=False,
            res_block_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            adm_in_channels=None,
    ):
        """
        UNet模型
        :param image_size: 输入图像的尺寸
        :param in_channels: 输入图像的通道数
        :param model_channels: 模型的基础通道数
        :param out_channels: 模型输出的通道数
        :param num_res_blocks: 每个下采样过程中残差块的数量
        :param attention_resolutions: 可以加入注意力机制的下采样过程
        :param dropout: 随机Dropout的概率
        :param channel_mult: UNet每层的通道乘数，主要是用来确定UNet每一层残差块的数量 [1, 2, 4, 4]
        :param conv_resample: 是否使用卷积操作进行下采样和上采样
        :param dims: 卷积神经网络的维数
        :param num_classes: 分类任务的类别数
        :param use_checkpoint: 是否使用检查点技术来减少内存使用
        :param use_fp16: 是否使用半精度浮点数（float16）进行计算
        :param use_bf16: 是否使用 BFloat16 进行计算
        :param num_heads: 注意力机制中的头数
        :param num_head_channels: 每个注意力头的通道数，可以根据num_head_channels计算num_heads
        :param num_heads_up_sample: 上采样过程中注意力机制的头数
        :param use_scale_shift_norm: 是否使用FiLM条件机制
        :param res_block_updown: 是否使用残差块进行下采样和上采样
        :param use_new_attention_order: 是否使用新的注意力计算顺序
        :param use_spatial_transformer: 是否使用Spatial Transformer
        :param transformer_depth: Spatial Transformer的深度
        :param context_dim: Spatial Transformer中的Key向量和Value向量
        :param n_embed: codebook中的向量维数
        :param legacy: 在分开QKV向量之前分割注意力头
        :param disable_self_attentions: 在每个分辨率级别上是否禁用自注意力机制
        :param num_attention_blocks: 每个分辨率级别上使用注意力机制的块数
        :param disable_middle_self_attn: 是否禁用中间块的自注意力机制
        :param use_linear_in_transformer: 在Spatial Transformer中是否使用线性层
        :param adm_in_channels: 用于顺序标签嵌入（sequential label embedding）的输入通道数
        """
        super(UNetModel, self).__init__()

        """
        性能平衡：在实验中，UNet 模型在不同层次上应用注意力机制时，发现适度使用注意力块可以显著提高模型的性能，而过多的注意力块会导致计算成本过高，性能提升有限
        最佳实践：通常在下采样和上采样的中间层次上应用注意力机制，这些层次的特征图尺寸较小，计算成本相对较低，同时可以有效捕捉全局信息
        """

        if use_spatial_transformer:
            assert context_dim is not None, "忘记在cross-attention中设置key向量和value向量"

        if context_dim is not None:
            assert use_spatial_transformer, "忘记在cross-attention中使用spatial transformer"
            """
            ListConfig: 配置列表
            """
            from omegaconf.listconfig import ListConfig

            if isinstance(context_dim, ListConfig):
                context_dim = list(context_dim)

        if num_heads_up_sample == -1:
            # 上采样过程中注意力机制的头数
            num_heads_up_sample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, '必须设置num_heads或num_head_channels'

        if num_head_channels == -1:
            assert num_heads != -1, '必须设置num_heads或num_head_channels'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            # UNet每一层的残差块的数量
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("num_res_blocks为int类型或作为列表/元组时，层数必须与channel_mult相同")
            # 如果num_res_blocks的类型不为int，自定义UNet每一层的残差块的数量
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # 当在UNet模型中禁用self-attention机制时
            # disable_self_attentions = [False, False, True,...]
            assert len(disable_self_attentions) == len(channel_mult), \
                "disable_self_attentions的层数必须与channel_mult的层数保持一致"

        if num_attention_blocks is not None:
            # UNet每一层的注意力块的数量
            assert len(num_attention_blocks) == len(self.num_res_blocks), "注意力块的层数必须等于残差块的层数"

            """
            map(): 将一个函数应用于一个或多个可迭代对象的每一个元素，并返回一个迭代器，这个迭代包含所有函数调用的结果
            
                example:
                    numbers = [1, 2, 3, 4, 5]
                    # 使用 map() 函数将每个元素平方
                    squared = map(lambda x: x ** 2, numbers)
                    # 将迭代器转换为列表
                    squared_list = list(squared)
                    print(squared_list)  # 输出：[1, 4, 9, 16, 25]
                    
            all(): 检查所有元素是否都为 True。如果所有元素都为 True，则返回 True；否则返回 False
            """
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)))), "每一层注意力块的数量必须小于等于残差块的数量"

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.bfloat16 if use_bf16 else self.dtype

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_up_sample = num_heads_up_sample
        self.predict_codebook_ids = n_embed is not None

        time_embedding_dim = model_channels * 4
        # 对时间进行编码
        """
        线性层
        Swish激活函数
        线性层
        """
        self.time_embedding = nn.Sequential(
            nn.Linear(in_features=model_channels, out_features=time_embedding_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_embedding_dim, out_features=time_embedding_dim)
        )

        if self.num_classes is not None:
            """
            离散的: 离散的标签
            连续的: 连续的标签
            顺序的: 顺序性的标签
            """
            if isinstance(self.num_classes, int):
                # 如果是整数，创建一个嵌入层，将类别 ID 映射到时间嵌入维度
                # 对标签进行编码
                self.label_embedding = nn.Embedding(num_classes, time_embedding_dim)
            elif self.num_classes == "continuous":
                # 如果是"continuous"，创建一个线性层，将连续的标签映射到时间嵌入维度
                self.label_embedding = nn.Linear(1, time_embedding_dim)
            elif self.num_classes == "sequential":
                # 如果是"sequential"，创建一个包含多个线性层和激活函数的序列，将顺序标签映射到时间嵌入维度
                assert adm_in_channels is not None, "标签的通道数不能为None"

                """
                线性层
                Swish激活函数
                线性层
                """
                self.label_embedding = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(in_features=adm_in_channels, out_features=time_embedding_dim),
                        nn.SiLU(),
                        nn.Linear(in_features=time_embedding_dim, out_features=time_embedding_dim),
                    )
                )
            else:
                raise ValueError()

        """
        self.input_blocks:
            卷积层: 调整输入数据的通道数
            [残差块, 注意力块, 残差块, 注意力块, 下采样残差块] * 3
            [残差块, 残差块]
        """
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbeddingSequential(conv_nd(dims, in_channels, model_channels, kernel_size=3, padding=1))
            ]
        )

        self._feature_size = model_channels

        """
        input_block_channels: [model_channels, mult_i * model_channels * 3 * 3, mult_{-1} * model_channels * 2]
        """
        input_block_channels = [model_channels]
        channels = model_channels
        # 下采样因子为1、2、4时，残差块后面可以跟着注意力块；当下采样因子为8时，残差块后面不再跟着注意力块
        ds = 1

        # level: 层数
        for level, mult in enumerate(channel_mult):
            # res_block: 每一层残差块的数量
            for res_block in range(self.num_res_blocks[level]):
                # 残差块
                layers = [
                    ResBlock(
                        channels=channels,
                        embedding_channels=time_embedding_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                ]

                channels = mult * model_channels

                # attention_resolutions: [4, 2, 1]
                """
                attention_resolutions: 在不同分辨率的特征图之间建立 attention 连接
                    在Transtomer的编码器和解码器中，输入序列会被划分为多个子序列，每个子序列中的词向量都会互相注意到，但不同子序列之间的词向量则不会相互影响
                    UNet 中的 attention_resolutions 技术会将编码器和解码器中的特征图进行拼接，并将其输入到一个注意力机制中
                """
                if ds in attention_resolutions:
                    # num_head_channels: 每个注意力头的通道数
                    if num_head_channels == -1:
                        # dim_head: 每个注意力头的维数
                        dim_head = channels // num_heads
                    else:
                        num_heads = channels // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = channels // num_heads if use_spatial_transformer else num_head_channels

                    if disable_self_attentions is not None:
                        disable_self_attention = disable_self_attentions[layers]
                    else:
                        disable_self_attention = False

                    # 每一层注意力块的数量必须小于等于残差块的数量
                    # 如果残差块的数量小于注意力块的数量
                    if num_attention_blocks is None or res_block < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                channels,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_up_sample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                channels,
                                n_heads=num_heads,
                                d_head=dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disable_self_attention,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                self.input_blocks.append(TimestepEmbeddingSequential(*layers))

                self._feature_size += channels
                input_block_channels.append(channels)

            if level != len(channel_mult) - 1:
                out_channels = channels

                self.input_blocks.append(
                    TimestepEmbeddingSequential(
                        ResBlock(
                            channels=channels,
                            embedding_channels=time_embedding_dim,
                            dropout=dropout,
                            out_channels=out_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if res_block_updown
                        else DownSample(
                            channels=channels, use_conv=conv_resample, dims=dims, out_channels=out_channels
                        )
                    )
                )

                channels = out_channels

                input_block_channels.append(channels)

                ds *= 2
                self._feature_size += channels

        if num_head_channels == -1:
            dim_head = channels // num_heads

        else:
            num_heads = channels // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = channels // num_heads if use_spatial_transformer else num_head_channels

        """
        self.middle_block:
            残差块
            注意力块
            残差块
        """
        self.middle_block = TimestepEmbeddingSequential(
            ResBlock(
                channels=channels,
                embedding_channels=time_embedding_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),

            AttentionBlock(
                channels=channels,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                in_channels=channels,
                n_heads=num_heads,
                d_head=dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),

            ResBlock(
                channels=channels,
                embedding_channels=time_embedding_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self._feature_size += channels

        """
        input_block_channels: [model_channels, mult_i * model_channels * 3 * 3, mult_{-1} * model_channels * 2]
        
        self.output_blocks:
            [残差块, 残差块, 残差块, 上采样残差块]
            [残差块, 注意力块, 残差块, 注意力块, 残差块, 注意力块, 上采样残差块] * 2
            [残差块, 注意力块, 残差块, 注意力块, 残差块, 注意力块]
        """
        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                input_channels = input_block_channels.pop()

                """
                channels + input_channels: 对应layers的编码器的输出和解码器的输入进行相加
                """
                layers = [
                    ResBlock(
                        channels=channels + input_channels,
                        embedding_channels=time_embedding_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                channels = model_channels * mult

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = channels // num_heads
                    else:
                        num_heads = channels // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = channels // num_heads if use_spatial_transformer else num_head_channels

                    if disable_self_attentions:
                        disable_self_attention = disable_self_attentions[level]
                    else:
                        disable_self_attention = False

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                channels=channels,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_up_sample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                in_channels=channels,
                                n_heads=num_heads,
                                d_head=dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disable_self_attention,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                if level and i == self.num_res_blocks[level]:
                    out_channels = channels
                    layers.append(
                        ResBlock(
                            channels=channels,
                            embedding_channels=time_embedding_dim,
                            dropout=dropout,
                            out_channels=out_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if res_block_updown
                        else UpSample(channels, conv_resample, dims=dims, out_channels=out_channels)
                    )

                    ds //= 2

                self.output_blocks.append(TimestepEmbeddingSequential(*layers))

                self._feature_size += channels

        """
        组归一化
        Swish激活函数
        将模型的参数归零
        """
        self.out = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, kernel_size=3, padding=1)),
        )

        """
        组归一化
        Swish激活函数
        """
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                group_norm(channels),
                conv_nd(dims, model_channels, n_embed, kernel_size=1),
            )

    def forward(self, x, time_steps=None, context=None, label=None, **kwargs):
        """

        :param x: 输入的噪声数据 query向量
        :param time_steps: 时间步
        :param context: 条件数据 key向量和value向量
        :param label: 类别
        :param kwargs:
        :return:
        """

        assert (label is not None) == (self.num_classes is not None), "如果当且仅当模型依赖条件数据，则必须指定label"

        # hs: UNet模型中的隐层向量
        hs = []

        time_embedding = time_step_embedding(time_steps=time_steps, dim=self.model_channels, repeat_only=False)

        """
        线性层
        Swish激活函数
        线性层
        """
        # time_embedding.shape: [batch_size, model_channels] -> [batch_size, time_embedding_dim]
        time_embedding = self.time_embedding(time_embedding)

        if self.num_classes is not None:
            assert label.shape[0] == x.shape[0], "类别的batch_size必须和输入数据的batch_size相同"

            """
            线性层
            Swish激活函数
            线性层
            """
            # label_embedding.shape: [batch_size, adm_in_channels] -> [batch_size, time_embedding_dim]
            time_embedding = time_embedding + self.label_embedding(label)

        """
        self.dtype: fp16 or bf16
        """
        h = x.type(self.dtype)

        """
        input_block_channels: [model_channels, mult_i * model_channels * 3 * 3, mult_{-1} * model_channels * 2]
        
        self.input_blocks:
            卷积层: 调整输入数据的通道数
            [残差块, 注意力块, 残差块, 注意力块, 下采样残差块] * 3
            [残差块, 残差块]
        """
        for module in self.input_blocks:
            h = module(h, time_embedding, context)
            hs.append(h)

        """
        残差块
        注意力块
        残差块
        """
        h = self.middle_block(h, time_embedding, context)

        """
        [残差块, 残差块, 残差块, 上采样残差块]
        [残差块, 注意力块, 残差块, 注意力块, 残差块, 注意力块, 上采样残差块] * 2
        [残差块, 注意力块, 残差块, 注意力块, 残差块, 注意力块]
        """
        for module in self.output_blocks:
            """
            解码器中每个残差块的输入通道数是根据编码器的输入通道数和解码器的输入通道数相加得到的
            """
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, time_embedding, context)

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
