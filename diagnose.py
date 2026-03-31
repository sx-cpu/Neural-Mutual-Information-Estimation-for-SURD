import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.MLP import MINE
from utils.diagnose.diagnose_mine import plot_mi_curve
from utils.diagnose.ground_truth import generate_gaussian_mi_data
from utils.seed import set_global_seed

if __name__ == "__main__":
    set_global_seed(42)
    Nt = 2 * 10**6
    rho = 0.5

    # load data
    os.makedirs("./data", exist_ok=True)
    formatted_Nt = "{:.0e}".format(Nt).replace("+0", "").replace("+", "")
    filepath = os.path.join("./data", f"gaussian_mi_data_{formatted_Nt}_{rho}.npy")

    if os.path.isfile(filepath):
        print("Loading saved gaussian data ...")
        data = np.load(filepath, allow_pickle=True).item()
    else:
        print("Generating gaussian data ...")
        data = generate_gaussian_mi_data(N=Nt, rho=rho)
        np.save(filepath, data, allow_pickle=True)

    X = data["X"]
    Y = data["Y"]
    true_mi = data["true_mi"]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    print("Data ready. True MI", true_mi)

    # hyperparameter
    batch_size = 65536
    epochs = 40
    lr = 1e-4
    ema_rate = 0.01
    lambda_reg = 0.1
    C_reg = 0.0
    window_size = 100

    # train
    mine = MINE(dim_x=1, dim_y=1, device="cuda")
    mine.fit_normalization(X, Y)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    mi_list = mine.train(loader, epochs=epochs)

    # estimate
    MI_value = mine.estimate(X, Y, batch_size=65536, use_window=False)
    print(f"MI results = {MI_value:.6f}")

    # save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"logs/gaussian/gaussian_N={formatted_Nt}_rho={rho}_{ts}.png"
    title = "MI estimation curve for groud truth"
    plot_mi_curve(mi_list, save_path, title, hline=true_mi)
