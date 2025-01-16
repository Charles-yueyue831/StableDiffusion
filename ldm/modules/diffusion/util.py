# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : PyCharm
# @Software : PyCharm

import numpy as np
import math
import torch


def make_ddim_timesteps(ddim_discrete_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    """
    根据给定的离散化方法生成 DDIM 采样的时间步
    :param ddim_discrete_method: 离散化方法
    :param num_ddim_timesteps: 待生成的 DDIM 时间步数量
    :param num_ddpm_timesteps: 原始的 DDPM 时间步数量
    :param verbose: 是否打印详细信息
    :return:
    """

    # 检查ddim_discr_method是否为'uniform'（均匀离散化方法）
    if ddim_discrete_method == 'uniform':
        # 确定均匀采样的步长
        c = num_ddpm_timesteps // num_ddim_timesteps
        """
        np.asarray(): 将list转换为numpy数组
        """
        # 均匀离散化的 DDIM 时间步
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))

    # 如果ddim_discr_method不是'uniform'，则检查是否为'quad'（二次离散化方法）
    elif ddim_discrete_method == 'quad':
        """
        np.linspace(): 生成一个等差数列
        
        example:
            num_ddpm_timesteps = 100
            num_ddim_timesteps = 10
            
            # 生成等差数列
            linspace = np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)
            
            # 将每个元素平方
            squared = linspace ** 2
            
            print(squared)
            
            >>> [ 0.          0.79527079  3.18108316  7.04900553 12.40041019 19.22580625
                 27.52509361 37.29817225 48.54504225 61.2867025 ]
        """
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discrete_method}"')

    # 如何理解ddim_timesteps + 1？
    steps_out = ddim_timesteps + 1

    if verbose:
        print(f'为 DDIM 采样器选择的时间步: {steps_out}')

    return steps_out


def make_ddim_sampling_parameters(alpha_cumprod, ddim_timesteps, eta, verbose=True):
    """
    选择alphas计算方差
    :param alpha_cumprod: alpha的累积乘积
    :param ddim_timesteps: DDIM采样步数
    :param eta: DDIM中用来控制方差的系数
    :param verbose: 是否打印详细信息
    :return:
    """
    alphas = alpha_cumprod[ddim_timesteps]
    alphas_prev = np.asarray([alpha_cumprod[0]] + alpha_cumprod[ddim_timesteps[:-1]].tolist())

    # DDIM中的方差是一个可以人为调节的超参数
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

    if verbose:
        print(f'DDIM采样器: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'\\eta的选定值: {eta}\nDDIM采样器的\\sigma表格 {sigmas}')

    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_time_steps, alpha_bar, max_beta=0.999):
    """
    定义（1-beta）随时间从t=[0,1]的累积乘积
    :param num_diffusion_time_steps: 要生成的beta数量
    :param alpha_bar: 一个lambda，它接受从0到1的参数t，并产生扩散过程中（1-beta）的累积乘积
    :param max_beta: 最大beta值
    :return:
    """
    betas = []

    for i in range(num_diffusion_time_steps):
        t1 = i / num_diffusion_time_steps
        t2 = (i + 1) / num_diffusion_time_steps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    扩散系数\beta的调度器
    :param schedule: 调度器方式
    :param n_timestep: 时间步的数量
    :param linear_start: \beta的起始值
    :param linear_end: \beta的结束值
    :param cosine_s:
    :return:
    """
    if schedule == "linear":
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)

    elif schedule == "cosine":
        time_steps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)

        alphas = time_steps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "squared_cos_cap_v2":
        return betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)

    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5

    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas.numpy()
