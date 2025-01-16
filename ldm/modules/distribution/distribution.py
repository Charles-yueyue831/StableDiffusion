# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Time : 2025/1/16 16:09
# @Software : PyCharm

import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        """

        :param parameters: 包含均值和对数方差
        :param deterministic: 分布是否是确定性的
        """
        """
        对角矩阵: 一个主对角线之外的元素皆为0的矩阵
        协方差的每个位置就代表横纵元素之间的相关性
        如果多元高斯分布的协方差矩阵是对角阵，则生成的数据各个维度之间相互独立
        """
        self.parameters = parameters
        self.mean, self.log_var = torch.chunk(parameters, 2, dim=1)

        # 将self.log_var的值限制在 - 30.0 到 20.0 的范围内，避免过大或过小的值
        self.log_var = torch.clamp(self.log_var, -30.0, 20.0)

        self.deterministic = deterministic
        # 方差
        self.var = torch.exp(self.log_var)
        # 标准差
        self.std = torch.exp(0.5 * self.log_var)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """
        重参数技巧
        :return:
        """
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        """
        KL散度
        :param other:
        :return:
        """
        if self.deterministic:
            return torch.Tensor([0.])


