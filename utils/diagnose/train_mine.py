import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_mine(
    model_cls, X, Y, batch_size=65536, K=5, lr=1e-4, num_iters=2000, device="cuda"
):
    """
    训练 MINE / ReMINE
    返回:
        mi_list: 每次迭代 MI 估计
        loss_list: 损失
        ma_list: exp(T_marg) 的 moving average (ReMINE 正则)
        mine: 训练后的模型
    """

    mine = model_cls().to(device)
    optimizer = torch.optim.Adam(mine.parameters(), lr=lr)

    X = X.to(device)
    Y = Y.to(device)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ma_et = None
    mi_list, loss_list, ma_list = [], [], []

    iterator = iter(loader)

    for it in tqdm(range(num_iters)):
        try:
            bx, by = next(iterator)
        except:
            iterator = iter(loader)
            bx, by = next(iterator)

        bx = bx.to(device)
        by = by.to(device)

        # 计算 T_joint
        T_joint = mine.net(bx, by)

        # 计算 T_marg
        perm = torch.randperm(len(Y))[: len(bx)]
        by_m = Y[perm].to(device)
        T_marg = mine.net(bx, by_m)

        # exp(T_marg)
        et = torch.exp(T_marg)

        # ReMINE 的 moving average 正则
        if ma_et is None:
            ma_et = et.mean()
        else:
            ma_et = 0.99 * ma_et + 0.01 * et.mean()

        # ReMINE Loss（论文公式）
        loss = -(T_joint.mean() - torch.log(ma_et + 1e-12))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        loss_list.append(loss.item())
        ma_list.append(ma_et.item())

        # 每次迭代估计 MI（不使用窗口）
        mi = T_joint.mean().item() - np.log(ma_et.item())
        mi_list.append(mi)

    return mi_list, loss_list, ma_list, mine
