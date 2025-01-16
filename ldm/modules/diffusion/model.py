# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import einops
from typing import Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import MemoryEfficientCrossAttention

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except Exception:
    XFORMERS_IS_AVAILABLE = False
    print("No module 'xformers'. Proceeding without it.")


def group_norm(in_channels, num_group=32):
    """
    组归一化
    :param in_channels: 输入通道数
    :param num_group: 组数
    :return:
    """
    return nn.GroupNorm(num_groups=num_group, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x):
    """
    Swish激活函数
    :param x:
    :return:
    """
    return x * torch.sigmoid(x)


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    """

    :param in_channels: 输入通道数
    :param attn_type: 注意力机制类型
    :param attn_kwargs:
    :return:
    """
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear",
                         "none"], f'未知的注意力机制类型: {attn_type}'

    if XFORMERS_IS_AVAILABLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"注意力机制类型: '{attn_type}'\t输入通道数: {in_channels}")

    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)

    elif attn_type == "vanilla-xformers":
        return MemoryEfficientAttnBlock(in_channels)

    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)

    elif attn_type == "none":
        return nn.Identity(in_channels)

    else:
        raise NotImplementedError()


class DownSample(nn.Module):
    def __init__(self, in_channels, with_conv):
        """
        下采样
        :param in_channels: 输入通道数
        :param with_conv: 是否使用卷积神经网络
        """
        super(DownSample, self).__init__()

        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2)

    def forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        if self.with_conv:
            """
            pad = (0, 1, 0, 1): (左, 右, 上, 下)
            """
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

            x = self.conv(x)

        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, with_conv):
        """
        上采样
        :param in_channels: 输入通道数
        :param with_conv: 是否使用卷积神经网络
        """
        super(UpSample, self).__init__()

        self.with_conv = with_conv

        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        """
        F.interpolate():
            example:
                x = torch.randn(1, 3, 4, 4)
                x_upsampled = F.interpolate(x, scale_factor=2.0, mode="nearest")
                
                >>> torch.Size([1, 3, 8, 8])
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, *, channels, out_channels, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        """
        VAE中的Encoder
        :param channels: 输入通道数
        :param out_channels: 输出通道数
        :param ch_mult: 通道数乘数，用于在不同分辨率层中逐步增加通道数
        :param num_res_blocks: 每个分辨率层中的残差块（Residual Block）数量
        :param attn_resolutions: 需要应用注意力机制（Attention Mechanism）的分辨率层
        :param dropout: Dropout率，用于正则化模型，防止过拟合
        :param resamp_with_conv: 是否使用卷积层进行上采样（Upsampling）和下采样（Downsampling）
        :param in_channels: 输入数据的通道数
        :param resolution: 输入数据的分辨率 例如，对于256x256的图像，resolution 为256
        :param z_channels: 潜在空间（Latent Space）的通道数
        :param double_z: 是否将潜在空间的通道数加倍
        :param use_linear_attn: 是否使用线性注意力机制
        :param attn_type: 注意力机制的类型
        :param ignore_kwargs:
        """
        super(Encoder, self).__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.channels = channels
        self.time_embedding_channels = 0
        # self.num_resolutions: 下采样的层数
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # 下采样
        self.conv_in = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.channels, kernel_size=3, padding=1)
        current_resolution = resolution

        # 如何理解in_channels_mult？
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_channels_mult = in_ch_mult

        block_in = channels * in_ch_mult[0]
        block_out = channels * ch_mult[0]

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            """
            in_ch_mult = (1, 1, 2, 4, 8)
            ch_mult = (1, 2, 4, 8)
            """
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         time_embedding_channels=self.time_embedding_channels, dropout=dropout))

                block_in = block_out

                if current_resolution in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            """
            down = nn.Module(): 创建一个 nn.Module 的实例
                down.block = block: 动态地为 down 添加属性 block
            """
            down = nn.Module()

            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                # 如果不是下采样的最后一层
                down.downsample = DownSample(block_in, resamp_with_conv)

                current_resolution = current_resolution // 2

            self.down.append(down)

        # middle
        self.mid = nn.Module()

        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       time_embedding_channels=self.time_embedding_channels, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       time_embedding_channels=self.time_embedding_channels, dropout=dropout)

        # end
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=2 * z_channels if double_z else z_channels,
                                  kernel_size=3, padding=1)

    def forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        # timestep embedding
        time_embedding = None

        # 下采样
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            # 不同的下采样因子
            for i_block in range(self.num_res_blocks):
                # 不同下采样层的残差块数量
                # 在下采样过程中残差块和注意力块交替出现
                h = self.down[i_level].block[i_block](hs[-1], time_embedding)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, time_embedding)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_embedding)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(self, *, channels, out_channels, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        """
        VAE中的Decoder
        :param channels: 输入通道数
        :param out_channels: 输出通道数
        :param ch_mult: 通道数乘数，用于在不同分辨率层中逐步增加通道数
        :param num_res_blocks: 每个分辨率层中的残差块（Residual Block）数量
        :param attn_resolutions: 需要应用注意力机制（Attention Mechanism）的分辨率层
        :param dropout: Dropout率，用于正则化模型，防止过拟合
        :param resamp_with_conv: 是否使用卷积层进行上采样（Upsampling）和下采样（Downsampling）
        :param in_channels: 输入数据的通道数
        :param resolution: 输入数据的分辨率 例如，对于256x256的图像，resolution 为256
        :param z_channels: 潜在空间（Latent Space）的通道数
        :param give_pre_end: 是否在解码器的末尾返回预激活的输出
        :param tanh_out: 是否在输出上应用 tanh 激活函数，通常用于将输出范围限制在 [-1, 1]
        :param use_linear_attn: 是否使用线性注意力机制
        :param attn_type: 注意力机制的类型
        :param ignore_kwargs:
        """
        super(Decoder, self).__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.channels = channels
        self.time_embedding_channels = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        in_ch_mult = (1,) + tuple(ch_mult)

        block_in = channels * ch_mult[self.num_resolutions - 1]

        curr_resolution = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_resolution, curr_resolution)

        print(f"隐向量z的shape: {self.z_shape} = {np.prod(self.z_shape)}")

        # z to block_in
        self.conv_in = nn.Conv2d(in_channels=z_channels, out_channels=block_in, kernel_size=3, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       time_embedding_channels=self.time_embedding_channels, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       time_embedding_channels=self.time_embedding_channels, dropout=dropout)

        # 上采样
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = channels * ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         time_embedding_channels=self.time_embedding_channels, dropout=dropout))

                block_in = block_out
                if curr_resolution in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()

            up.block = block
            up.attn = attn

            if i_level != 0:
                up.up_sample = UpSample(block_in, resamp_with_conv)
                curr_resolution = curr_resolution * 2

            self.up.insert(0, up)

        # end
        self.norm_out = swish(block_in)
        self.conv_out = nn.Conv2d(in_channels=block_in, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        """

        :param z: 隐层向量
        :return:
        """

        # timestep embedding
        time_embedding = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, time_embedding)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_embedding)

        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            # 在上采样的过程中，残差块和注意力块交替出现
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, time_embedding)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)

        if self.tanh_out:
            h = torch.tanh(h)

        return h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, time_embedding_channels=512):
        """
        残差块
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param conv_shortcut: 是否使用卷积进行shortcut
        :param dropout: Dropout率，用于正则化模型，防止过拟合
        :param time_embedding_channels: 时间嵌入的通道数
        """
        super(ResnetBlock, self).__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if time_embedding_channels > 0:
            self.time_embedding_proj = nn.Linear(in_features=time_embedding_channels, out_features=out_channels)

        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                                     padding=1)

            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x, time_embedding):
        """

        :param x: 输入数据
        :param time_embedding: 时间嵌入
        :return:
        """
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        if time_embedding is not None:
            h = h + self.temb_proj(swish(time_embedding))[:, :, None, None]

        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        """
        注意力机制
        :param in_channels: 输入通道数
        """
        super(AttnBlock, self).__init__()

        self.in_channels = in_channels

        self.norm = group_norm(in_channels)

        self.q = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.k = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.v = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        """

        :param x: 输入数据
        :return:
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        """
        [batch_size, height*width, channel]相当于NLP中的[batch_size, seq_len, num_dim]
        """
        q = einops.rearrange(q, 'b c h w -> b (h w) c')
        k = einops.rearrange(k, 'b c h w -> b c (h w)')
        v = einops.rearrange(v, 'b c h w -> b c (h w)')

        # 计算注意力权重
        w_ = torch.einsum('b i c, b c j -> b i j', q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        h_ = torch.einsum('b c i, b i j -> b c j', v, w_)

        h_ = einops.rearrange(h_, 'b c (h w) -> b c h w', h=h, w=w)

        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    def __init__(self, in_channels):
        """

        :param in_channels: 输入通道数
        """
        super(MemoryEfficientAttnBlock, self).__init__()

        self.in_channels = in_channels

        self.norm = group_norm(in_channels)

        self.q = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.k = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.v = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        """
        map(): 将一个函数应用于一个或多个可迭代对象的每个元素，并返回一个迭代器
        
        example:
            numbers = [1, 2, 3, 4, 5]
    
            # 使用 map() 函数将每个元素平方
            squared = map(lambda x: x ** 2, numbers)
            
            # 将迭代器转换为列表
            squared_list = list(squared)
            
            print(squared_list)  # 输出：[1, 4, 9, 16, 25]
        """
        q, k, v = map(lambda x: einops.rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], 1, c)
            .permute(0, 2, 1, 3)
            .reshape(b * 1, t.shape[1], c)
            .contiguous(),
            (q, k, v),
        )

        """
        使用 xformers 库中的 memory_efficient_attention 函数来计算高效的注意力机制
            xformers 是一个高性能的 Transformer 组件库，特别优化了内存使用和计算效率
            
        xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op):
            attn_bias: 用于在注意力计算中添加偏置。通常用于实现 masked attention
            op: 指定使用的注意力操作
                xformers.ops.fmha.cutlass.FwOp：使用 Cutlass 库实现的前向传播操作，适用于 NVIDIA GPUs
                xformers.ops.fmha.cutlass.BwOp：使用 Cutlass 库实现的反向传播操作，适用于 NVIDIA GPUs
                xformers.ops.fmha.flash.FwOp：使用 FlashAttention 库实现的前向传播操作，适用于 NVIDIA GPUs
                xformers.ops.fmha.flash.BwOp：使用 FlashAttention 库实现的反向传播操作，适用于 NVIDIA GPUs
                xformers.ops.fmha.triton.FwOp：使用 Triton 库实现的前向传播操作，适用于 NVIDIA GPUs
                xformers.ops.fmha.triton.BwOp：使用 Triton 库实现的反向传播操作，适用于 NVIDIA GPUs
        """
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, 1, out.shape[1], c)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], c)
        )

        out = einops.rearrange(out, 'b (h w) c -> b c h w', b=b, h=h, w=w, c=c)

        out = self.proj_out(out)
        return x + out


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None):
        """

        :param x: query向量
        :param context: key向量和value向量
        :param mask: 掩码向量
        :return:
        """
        b, c, h, w = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        out = super().forward(x, context=context, mask=mask)
        out = einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)

        return x + out
