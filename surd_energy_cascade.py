import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import utils.surd as surd

# Configure matplotlib to use LaTeX for text rendering and set font size
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 22})

if __name__ == "__main__":
    # configs and datasets
    fname = "./data/energy_cascade_signals.mat"
    data = loadmat(fname)
    X = data["X"]
    nvars = X.shape[0]
    Nt = X.shape[1]
    nbins = 70
    nlags = np.array([1, 19, 11, 6])
    input_vars = [1, 2, 3, 4]

    q1, q2, q3, q4 = X
    data_map = {1: q1, 2: q2, 3: q3, 4: q4}

    information_flux = {}
    Rd_results = {}
    Sy_results = {}
    MI_results = {}
    info_leak_results = {}
    for target_var in range(1, nvars + 1):
        Y = data_map[target_var][nlags[target_var - 1] :]
        Y = Y.reshape(-1)

        # get the latest file
        folder = "results/MI_results"
        pattern = os.path.join(
            folder, f"energycascade_MI_results_target_{target_var}_*.npy"
        )
        files = glob.glob(pattern)
        if len(files) == 0:
            raise FileNotFoundError(f"No files found matching: {pattern}")

        latest_file = max(files, key=os.path.getmtime)
        print("loading latest MI file:", latest_file)

        MI = np.load(latest_file, allow_pickle=True).item()
        I_R, I_S, MI_out = surd.surd_global(MI, n_vars=4)

        # compute leak
        _, _, info_leak = surd.compute_info_leak(Y, MI, bins=nbins)

        surd.nice_print(I_R, I_S, MI, info_leak)
        print("\n")

        # Save the results
        (
            Rd_results[target_var],
            Sy_results[target_var],
            MI_results[target_var],
            info_leak_results[target_var],
        ) = I_R, I_S, MI, info_leak

    fig, axs = plt.subplots(
        nvars,
        2,
        figsize=(10, 2.3 * nvars),
        gridspec_kw={"width_ratios": [nvars * 14, 1]},
    )
    plt.rcParams.update({"font.size": 18})
    for i in range(nvars):
        information_flux[i + 1] = surd.plot_nlabels(
            Rd_results[i + 1],
            Sy_results[i + 1],
            info_leak_results[i + 1],
            axs[i, :],
            nvars,
            nlabels=12,
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i + 1}}} / I \\left(\\Sigma_{i + 1}^+ ; \\mathrm{{\\mathbf{{\\Sigma}}}} \\right)$",
            pad=10,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i + 1}}}}}{{H \\left(\\Sigma_{i + 1} \\right)}}$",
            pad=17,
        )
        axs[i, 1].set_yticks([0, 1])
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=16,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )

    # Show the results
    plt.tight_layout(w_pad=-15, h_pad=-0.1)
    plot_save_path = "./results/surd_results/energy_cascade.png"
    plt.savefig(plot_save_path)
    plt.show()
