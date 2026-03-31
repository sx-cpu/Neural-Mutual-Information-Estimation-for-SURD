import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def plot_mi_curve(mi_list, save_path=None, title="MI estimation curve", hline=None):
    plt.figure(figsize=(15, 8))
    plt.plot(mi_list, label="Estimated MI")

    if hline is not None:
        plt.axhline(y=hline, color="r", linestyle="--", label="True MI")
    plt.xlabel("Training iteration")
    plt.ylabel("MI")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def diagnose_mine(mine, X, Y, batch_size=65536, K=5, true_mi=None):
    """
    mine: 你的 MINE 模型
    X, Y: 原始数据 (未标准化的)
    true_mi: 如果是合成数据，可提供 ground truth
    """

    device = mine.device
    X = X.to(device)
    Y = Y.to(device)

    print("\n================= MINE 诊断报告 =================")

    # ---------------------------------------------------------
    # (1) 归一化并计算 T/T_marg 统计
    # ---------------------------------------------------------
    Xn, Yn = mine.normalize(X, Y)
    loader = DataLoader(TensorDataset(Xn, Yn), batch_size=batch_size, shuffle=False)

    T_vals = []
    Tm_vals = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)

            t = mine.net(bx, by)
            T_vals.append(t.cpu())

            perm = torch.randperm(len(Yn))[: len(by)]
            by_m = Yn[perm].to(device)
            tm = mine.net(bx, by_m)
            Tm_vals.append(tm.cpu())

    T_all = torch.cat(T_vals).numpy()
    Tm_all = torch.cat(Tm_vals).numpy()

    print(f"T mean           = {T_all.mean():.6f}, std = {T_all.std():.6f}")
    print(f"T_marg mean      = {Tm_all.mean():.6f}, std = {Tm_all.std():.6f}")
    print(f"exp(T_marg) mean = {np.exp(Tm_all).mean():.6f}")

    # collapse 检测
    if T_all.mean() < -30:
        print("⚠ [CRITICAL] T 已坍缩到负无穷（模型 collapse）")

    if np.exp(Tm_all).mean() > 1e6:
        print("⚠ [WARNING]  exp(T_marg) 巨大，可能是虚高!!")

    # ---------------------------------------------------------
    # (2) ReMINE buffer 查看是否收敛
    # ---------------------------------------------------------
    if len(mine._stat_buffer) > 0:
        t_buf, et_buf = zip(*mine._stat_buffer)
        t_buf = np.array(t_buf)
        et_buf = np.array(et_buf)

        mi_buf = np.mean(t_buf) - np.log(np.mean(et_buf) + 1e-12)

        print("\n—— Window Buffer（micro-averaging）——")
        print(f"buffer size      = {len(t_buf)}")
        print(f"buffer t_mean    = {t_buf.mean():.6f}")
        print(f"buffer log(et)   = {np.log(et_buf.mean() + 1e-12):.6f}")
        print(f"Window MI        = {mi_buf:.6f}")
    else:
        print("\n⚠ buffer 为空（训练可能太短）")

    # ---------------------------------------------------------
    # (3) 多次估计稳定性
    # ---------------------------------------------------------
    mis = []
    for _ in range(K):
        mi_val = mine.estimate(X, Y, batch_size=batch_size, K=1, use_window=False)
        mis.append(mi_val)

    print("\n—— 多次完整估计 ——")
    print(f"MI mean = {np.mean(mis):.6f}, std = {np.std(mis):.6f}")

    # ---------------------------------------------------------
    # (4) 如果提供真实 MI，则显示对比
    # ---------------------------------------------------------
    if true_mi is not None:
        print(f"\nGround truth MI  = {true_mi:.6f}")
        print(f"Estimation error = {np.mean(mis) - true_mi:.6f}")

    print("\n================= 诊断结束 =================\n")

    # ---------------------------------------------------------
    # (5) 绘制 T/T_marg 分布图
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(T_all, bins=50)
    plt.title("T distribution")

    plt.subplot(1, 2, 2)
    plt.hist(Tm_all, bins=50)
    plt.title("T_marg distribution")

    plt.show()
