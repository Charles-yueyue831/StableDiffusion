# Stable Diffusion Version 2

https://github.com/Stability-AI/stablediffusion 仓库中代码的注解版本

## VAE

生成模型的难题：<font color="orange">**判断生成分布与真实分布的相似度**</font>

$p(X)=\sum_{Z}p(X|Z)p(Z)$: $p(X|Z)$描述一个由Z生成X的模型，假设 $p(Z|X)$ 是正态分布

使用神经网络拟合出专属于 $X_k$ 的正态分布 $P(Z|X_k)$ 的均值和方差: $μ_k=f_1(X_k),logσ^2=f_2(X_k)$

在知道专属于 $X_k$ 的正态分布 $P(Z|X_k)$ 的均值和方差后，可以从 $P(Z|X_k)$ 中采样得到 $Z_k$ ，经过生成器得到 $\hat{X}_k=g(Z_k)$ ，并最小化 $D(\hat{X}_k,X_k)^2$

由于 $Z_k$ 是从 $P(Z|X_k)$ 中采样得到的，会引入一定的噪声

为了保证模型的生成能力，VAE还让所有的 $p(Z|X)$ 尽量满足标准正态分布 $to$ $p(Z)=\sum_{X}p(Z|X)p(X)=\sum_X \mathcal{N}(0,1)p(X)=\mathcal{N}(0,1)\sum_Xp(X)
= \mathcal{N}(0,1)$

VAE的损失函数: 采样样本和生成样本之间的重构损失 $D(\hat{X}_k,X_k)^2$: $p(Z|X)$与标准正态分布之间的KL散度 $KL(\mathcal{N}(μ,σ^2)‖\mathcal{N}(0,1))$

## Reference

1、指数移动平均（EMA）: https://blog.csdn.net/weixin_43135178/article/details/122147538
