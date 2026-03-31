import numpy as np


def generate_gaussian_mi_data(N=200000, rho=0.8, dim=1, seed=None):
    """
    使用 NumPy 生成具有真实互信息的高斯数据
    X, Y ~ N(0,1)，相关系数为 rho
    true_mi = -0.5 * dim * log(1 - rho^2)

    参数：
    - N: 样本数
    - rho: 相关系数
    - dim: 维度
    - seed: 随机种子

    返回：
    X, Y, true_mi
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(N, dim)
    eps = np.random.randn(N, dim)
    Y = rho * X + np.sqrt(1 - rho**2) * eps

    true_mi = -0.5 * dim * np.log(1 - rho**2)

    return {"X": X, "Y": Y, "true_mi": true_mi}
