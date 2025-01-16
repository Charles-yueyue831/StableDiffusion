# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/13 15:04
# @Software : PyCharm

from inspect import isfunction
from typing import Optional, Any
import os
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ldm import checkpoint

try:
    """
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
        index-url: 从PyTorch官方提供的预编译wheel文件的URL获取包
        
    xformers 是一个用于高效 Transformer 模型实现的库，旨在提供高性能和可扩展的 Transformer 组件
    """
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except Exception as e:
    XFORMERS_IS_AVAILABLE = False

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def group_normalize(in_channels):
    """
    组归一化
    :param in_channels: 输入通道数
    :return:
    """
    """
    nn.GroupNorm():
        affine=True: 是否进行仿射变换
            在归一化之后，数据的均值为零，方差为一。然而，这种标准化的数据可能不总是最适合特定的任务
            为了增加模型的灵活性，可学习的仿射变换允许模型学习一组参数，以调整归一化后的数据，使其能够更好地拟合数据
    """
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def default(val, d):
    """
    判断val的存在与否决定val的值
    :param val: 原始数据
    :param d: 备用数据
    :return:
    """
    if val is not None:
        return val

    return d() if isfunction(d) else d


def zero_module(module):
    """

    :param module: 神经网络模块
    :return:
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        GEGLU: 带有门控机制的GELU激活函数
        :param dim_in: 输入数据维数
        :param dim_out: 输出数据维数
        """
        super(GEGLU, self).__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """
        GEGLU: GELU(x_{gate})*x_{linear}
        :param x: 输入数据
        :return:
        """
        x, gate = self.proj(x).chunk(2, dim=-1)

        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        """

        :param dim: 输入向量维数
        :param dim_out: 输出向量维数
        :param mult: 乘数
        :param glu: 是否为GEGLU激活函数
        :param dropout: 随机Dropout的概率
        """
        super(FeedForward, self).__init__()

        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        """
        SpatialSelfAttention机制
        :param in_channels: 输入通道数
        """
        super(SpatialSelfAttention, self).__init__()

        self.in_channels = in_channels

        self.norm = group_normalize(in_channels)

        self.q = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.project_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_x = self.norm(x)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape

        q = einops.rearrange(q, 'b c h w -> b (h w) c')
        k = einops.rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij, bjk -> bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = einops.rearrange(v, 'b c h w -> b c (h w)')
        w_ = einops.rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = einops.rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class SpatialTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True):
        """
        SpatialTransformer
        :param in_channels: 输入通道数
        :param n_heads: 注意力头的数量
        :param d_head: 每个注意力头的维度
        :param depth: 如何理解depth？
        :param dropout: 随机Dropout的概率
        :param context_dim: 上下文内容的维度
        :param disable_self_attn: 是否禁用自注意力机制
        :param use_linear: 在映射时是否使用线性层
        :param use_checkpoint: 是否使用检查点
        """
        super(SpatialTransformer, self).__init__()

        """
        多头注意力机制:
            self.embed_dim = embed_dim  # 嵌入维度
            self.num_heads = num_heads  # 头的数量
            self.head_dim = embed_dim // num_heads  # 每个头的维度
        """

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = group_normalize(in_channels)

        if not use_linear:
            self.project_in = nn.Conv2d(in_channels=in_channels, out_channels=inner_dim, kernel_size=1)
        else:
            self.project_in = nn.Linear(in_features=in_channels, out_features=inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
             for d in range(depth)]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(in_channels=inner_dim, out_channels=in_channels, kernel_size=1))
        else:
            self.proj_out = zero_module(nn.Linear(in_features=in_channels, out_features=inner_dim))

        self.use_linear = use_linear

    def forward(self, x, context=None):
        """

        :param x: 输入数据（query向量）
        :param context: 上下文数据（key向量、value向量）
        :return:
        """
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape

        x_in = x
        # group normalization
        x = self.norm(x)

        if not self.use_linear:
            x = self.proj_in(x)

        """
        图像中的(batch_size, height*width, channel)对应文本中的(batch_size, seq_len, num_embedding)
        """
        x = einops.rearrange(x, 'b c h w -> b (h w) c').contiguous()

        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])

        if self.use_linear:
            x = self.proj_out(x)

        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in


class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        """

        :param query_dim: query向量的维数
        :param context_dim: key向量和value向量的维数
        :param heads: 注意力头的数量
        :param dim_head: 注意力头的维数
        :param dropout: 随机Dropout的概率
        """
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads

        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(in_features=query_dim, out_features=inner_dim, bias=False)
        self.to_k = nn.Linear(in_features=context_dim, out_features=inner_dim, bias=False)
        self.to_v = nn.Linear(in_features=context_dim, out_features=inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """

        :param x: 输入数据（query向量）
        :param context: 上下文数据（key向量、value向量）
        :param mask: 掩码向量
        :return:
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if _ATTN_PRECISION == "fp32":
            """
            torch.autocast(enabled=False, device_type='cuda'):
                enabled: 是否启用自动混合精度。如果 enabled=True，则在上下文管理器内部，某些操作将自动转换为较低精度的浮点数
                         如果 enabled=False，则上下文管理器内部的运算不会进行自动混合精度转换，所有操作将使用默认的精度
            """
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                similarity = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            similarity = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            """
            rearrange(mask, 'b ... -> b (...)'): 如果 mask 最初的形状是 (b, h, w)，经过 rearrange 后，它将变为 (b, h * w)
            """
            mask = einops.rearrange(mask, 'b ... -> b (...)')

            """
            torch.finfo(dtype):  返回一个 torch.finfo 对象
                dtype: 指定的浮点数数据类型，如 torch.float32、torch.float64、torch.float16 或 torch.bfloat16
            """
            max_negative_value = -torch.finfo(sim.dtype).max

            """
            repeat(tensor, pattern, **kwargs): 于重复张量的某些维度
                tensor: 要操作的张量
                pattern: 一个字符串，描述如何重复张量的维度
            """
            # shape从(b, j)变成(b*h, 1, j)
            mask = einops.repeat(mask, 'b j -> (b h) () j', h=h)

            """
            tensor.masked_fill_(mask, value): PyTorch 中的一个 inplace 方法，用于将张量中某些位置的元素替换为指定的值
                mask: 指示哪些位置需要被修改
                value: 用于替换的值
                
                example:
                    # 创建一个形状为 (2, 3) 的张量
                    sim = torch.tensor([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0]])
                    
                    # 创建一个形状为 (2, 3) 的布尔张量
                    mask = torch.tensor([[True, False, True],
                                         [False, True, False]])
                    
                    # 设置 max_neg_value
                    max_neg_value = -1e9
                    
                    # 使用 masked_fill_ 方法修改 sim
                    sim.masked_fill_(~mask, max_neg_value)
                    
                    >>>tensor([[ 1.0000e+00, -1.0000e+09,  3.0000e+00],
                               [-1.0000e+09,  5.0000e+00, -1.0000e+09]])
                ~mask: 取反操作
            """
            similarity.masked_fill_(~mask, max_negative_value)

        similarity = similarity.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', similarity, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        """
        MemoryEfficientCrossAttention
        :param query_dim: query向量的维数
        :param context_dim: key向量和value向量的维数
        :param heads: 注意力头的数量
        :param dim_head: 注意力头的维数
        :param dropout: 随机Dropout的概率
        """
        super(MemoryEfficientCrossAttention, self).__init__()

        print(f"设置{self.__class__.__name__}. Query向量的维数{query_dim}, Key向量和Value向量的维数{context_dim}")

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(in_features=query_dim, out_features=inner_dim, bias=False)
        self.to_k = nn.Linear(in_features=context_dim, out_features=inner_dim, bias=False)
        self.to_v = nn.Linear(in_features=context_dim, out_features=inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=query_dim),
            nn.Dropout(dropout)
        )

        """
        Optional[Any]
            Optional[T] 等价于 Union[T, None]，即该属性可以是类型 T 或 None
            Any 类型的变量可以是任何值，不进行类型检
        """
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        """

        :param x: 输入数据（query向量）
        :param context: 上下文数据（key向量、value向量）
        :param mask: 掩码向量
        :return:
        """
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # q.shape = [batch_size, head_channels * heads, dim_head]
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        """
        xformers.ops.memory_efficient_attention(): 实现了高效的注意力机制，特别优化了内存使用，适用于处理长序列和大批量数据
            attn_bias: 用于在注意力计算中添加偏置
        """
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if mask is not None:
            raise NotImplementedError

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        """

        :param dim: query向量
        :param n_heads: 注意力头的数量
        :param d_head: 注意力头的维数
        :param dropout: 随机Dropout的概率
        :param context_dim: key向量和value向量的维数
        :param gated_ff: 是否为GEGLU激活函数
        :param checkpoint: checkpoint()方法，梯度检查点
        :param disable_self_attn: 是否禁用自注意力机制
        """
        super(BasicTransformerBlock, self).__init__()

        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILABLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES, "attn_mode必须在ATTENTION_MODES内部"

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)

        self.feed_forward = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)

        """
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)
        """
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def _forward(self, x, context=None):
        """

        :param x: query向量
        :param context: key向量和value向量
        :return:
        """
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.feed_forward(self.norm3(x)) + x
        return x

    def forward(self, x, context=None):
        """

        :param x: query向量
        :param context: key向量和value向量
        :return:
        """
        return checkpoint(func=self._forward, x=(x, context), params=self.parameters(), flag=self.checkpoint)
