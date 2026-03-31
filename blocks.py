import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import utils.analytic_eqs as cases
import utils.datasets as proc
from model.MLP import MINE
from utils.diagnose.diagnose_mine import plot_mi_curve
from utils.seed import set_global_seed

# ----------------------------------------
# main
# ----------------------------------------
if __name__ == "__main__":
    set_global_seed(42)

    # -----------------------------
    # configs
    # -----------------------------
    Nt = 2 * 10**6
    nlag = 1
    transient = 10000
    samples = Nt - transient

    block = cases.synergistic_collider
    target_var = 3
    input_vars = [1, 2, 3]

    block_name = block.__name__
    os.makedirs("./data", exist_ok=True)
    formatted_Nt = "{:.0e}".format(Nt).replace("+0", "").replace("+", "")
    filepath = os.path.join("./data", f"{block_name}_Nt_{formatted_Nt}.npy")

    # -----------------------------
    #  datasets
    # -----------------------------
    if os.path.isfile(filepath):
        print(f"Loading saved {block_name} data ...")
        q1, q2, q3 = np.load(filepath, allow_pickle=True)
    else:
        print(f"Generating {block_name} data ...")
        qs = block(Nt)
        q1, q2, q3 = [q[transient:] for q in qs]
        np.save(filepath, [q1, q2, q3])
        print(f"Saved {block_name} data to", filepath)

    data_map = {1: q1, 2: q2, 3: q3}

    # -----------------------------
    #  comb
    # -----------------------------
    subsets = proc.all_subsets(input_vars)  # 例如 [(1,), (2,), (1,2)]
    print("Variable sets to evaluate:", subsets)

    # -----------------------------
    #  all results
    # -----------------------------
    MI_results = {}

    # -----------------------------
    # hyperparameter
    # -----------------------------
    batch_size = 65536
    epochs = 35
    ema_rate = 0.01
    lambda_reg = 0.1
    C_reg = 0.0
    window_size = 500
    batch_size = 65536
    lr = 3e-4

    for subset in subsets:
        print(
            f"\n=== Training MINE for inputs {subset} → target {target_var}[+{nlag}] ==="
        )

        # build full X,Y tensors first
        X_list = [data_map[v][:-nlag] for v in subset]
        X = np.vstack(X_list).T
        X = torch.tensor(X, dtype=torch.float32)

        Y = data_map[target_var][nlag:]
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
        save_path = f"logs/{block_name}/tar{target_var}_{block_name}_{subset}_{ts0}.png"
        title = f"MI estimation curve for {block_name} ({subset} to {target_var})"
        plot_mi_curve(mi_list, save_path, title)

        # estimate MI
        MI_value = mine.estimate(X, Y, batch_size=65536)
        MI_results[subset] = float(MI_value)

        print(f"MI{subset} = {MI_value:.6f}")

    # -----------------------------
    #  save results
    # -----------------------------

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath = (
        f"./results/MI_results/{block_name}_MI_results_target_{target_var}_{ts}.npy"
    )
    np.save(savepath, MI_results)
    print(f"\nSaved MI results to {savepath}")

    print("\nFinal MI results:")
    for k, v in MI_results.items():
        print(k, ":", v)
