# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        """

        :param model: 应用 EMA 的模型
        :param decay: EMA 的衰减率，默认为 0.9999
        :param use_num_updates: 是否使用更新次数来调整衰减率
        """
        super(LitEma, self).__init__()

        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        """
        name2s_name = {原本的模型参数名称: 移除.之后的模型参数名称}
        """
        self.name2s_name = {}

        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates',
                             torch.tensor(0, dtype=torch.int) if use_num_updates else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace('.', '')
                self.name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        """
        重置更新次数
        :return:
        """
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            # 根据更新次数动态调整衰减率，取初始衰减率和新计算的衰减率中的较小值
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            """
            model.named_parameters(): 迭代器
                'generator' object is not subscriptable
            """
            model_param = dict(model.named_parameters())

            """
            self.named_buffers(): 包含缓冲区的名称和对应的张量
            """
            shadow_params = dict(self.named_buffers())

            for key in model_param:
                if model_param[key].requires_grad:
                    # s_name和key之间一一对应
                    s_name = self.name2s_name[key]

                    shadow_params[s_name] = shadow_params[s_name].type_as(model_param[key])
                    """
                    EMA: v_t=\beta\cdot v_{t-1}+(1-\beta)\cdot\theta_t
                            v_{t-1}: 前 t -1 次更新的所有参数平均值，也称为影子权重shadow weights
                            \theta_t: t 时刻的模型权重weights
                            \beta: 在代码中一般写为decay，一般设为0.9-0.999
                            
                    sub_(): in-place操作
                        shadow_params[s_name].sub_(one_minus_decay * (shadow_params[s_name] - model_param[key])) = 
                            (1 - decay) * model_param[key] + decay * shadow_params[s_name]
                    """
                    shadow_params[s_name].sub_(one_minus_decay * (shadow_params[s_name] - model_param[key]))
                else:
                    assert key not in self.name2s_name

    def store(self, parameters):
        """
        存储模型的参数
        :param parameters: 模型参数
        :return:
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        恢复使用`store`方法存储的参数
        在不影响原始优化过程的情况下，使用EMA参数验证模型非常有用
        :param parameters:
        :return:
        """
        for collected_param, parameter in zip(self.collected_params, parameters):
            parameter.data.copy_(collected_param.data)
