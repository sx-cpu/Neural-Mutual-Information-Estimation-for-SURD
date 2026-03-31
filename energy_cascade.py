import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

import utils.datasets as proc
from model.MLP import MINE
from utils.diagnose.diagnose_mine import plot_mi_curve

# Configure matplotlib to use LaTeX for text rendering and set font size
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 22})


# ----------------------------------------
# main
# ----------------------------------------
if __name__ == "__main__":
    # configs and datasets
    fname = "./data/energy_cascade_signals.mat"
    data = loadmat(fname)
    X = data["X"]
    nvars = X.shape[0]
    Nt = X.shape[1]
    nbins = 10
    nlags = np.array([1, 19, 11, 6])
    target_var = 4
    input_vars = [1, 2, 3, 4]

    q1, q2, q3, q4 = X
    data_map = {1: q1, 2: q2, 3: q3, 4: q4}

    # comb
    subsets = proc.all_subsets(input_vars)
    print("Variable sets to evaluate:", subsets)

    # all results
    MI_results = {}

    # hyperparameter
    batch_size = 65536
    epochs = 1000
    lr = 2e-4
    ema_rate = 0.01
    lambda_reg = 0.1
    C_reg = 0.0
    window_size = 100
    batch_size = 65536

    for subset in subsets:
        print(
            f"\n=== Training MINE for inputs {subset} → target {target_var}[+{nlags[target_var - 1]}] ==="
        )

        # build full X,Y tensors first
        X_list = [data_map[v][: -nlags[target_var - 1]] for v in subset]
        X = np.vstack(X_list).T
        X = torch.tensor(X, dtype=torch.float32)

        Y = data_map[target_var][nlags[target_var - 1] :]
        Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

        # 1) fit normalization on full training data BEFORE creating dataloader (or do it now)
        mine = MINE(
            dim_x=X.shape[1],
            dim_y=1,
            lr=lr,
            ema_rate=ema_rate,
            lambda_reg=lambda_reg,
            C_reg=C_reg,
            window_size=window_size,
            device="cuda",
        )
        mine.fit_normalization(X, Y)  # <-- IMPORTANT: full-data fit

        # 2) now create dataset/loader and train
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mi_list = mine.train(loader, epochs=epochs)

        ts0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            f"logs/energy_cascade/tar{target_var}_energycascade_{subset}_{ts0}.png"
        )
        title = f"MI estimation curve for energy cascade ({subset} to {target_var})"
        plot_mi_curve(mi_list, save_path, title)

        # estimate MI
        MI_value = mine.estimate(X, Y, batch_size=65536)
        MI_results[subset] = float(MI_value)

        print(f"MI{subset} = {MI_value:.6f}")

    # save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath = (
        f"./results/MI_results/energycascade_MI_results_target_{target_var}_{ts}.npy"
    )
    np.save(savepath, MI_results)
    print(f"\nSaved MI results to {savepath}")

    print("\nFinal MI results:")
    for k, v in MI_results.items():
        print(k, ":", v)
