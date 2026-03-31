import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import utils.surd as surd

# Configure matplotlib to use LaTeX for text rendering and set font size
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 22})

if __name__ == "__main__":
    # information
    block = "synergistic_collider"
    lag = 1
    Nt = 2 * 10**6
    bins = 100
    nvars = 3

    information_flux = {}
    Rd_results = {}
    Sy_results = {}
    MI_results = {}
    info_leak_results = {}

    # file path
    formatted_Nt = "{:.0e}".format(Nt).replace("+0", "").replace("+", "")
    filepath = os.path.join("./data", f"{block}_Nt_{formatted_Nt}.npy")

    for target_var in range(1, nvars + 1):
        # datasets
        q1, q2, q3 = np.load(filepath, allow_pickle=True)
        data_map = {1: q1, 2: q2, 3: q3}
        Y = data_map[target_var][lag:]
        Y = Y.reshape(-1)

        # get the latest file
        folder = "results/MI_results"
        pattern = os.path.join(folder, f"{block}_MI_results_target_{target_var}_*.npy")
        files = glob.glob(pattern)
        if len(files) == 0:
            raise FileNotFoundError(f"No files found matching: {pattern}")

        latest_file = max(files, key=os.path.getmtime)
        print("loading latest MI file:", latest_file)

        MI = np.load(latest_file, allow_pickle=True).item()
        I_R, I_S, MI_out = surd.surd_global(MI, n_vars=3)

        # compute leak
        _, _, info_leak = surd.compute_info_leak(Y, MI, bins=bins)

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
        nvars, 2, figsize=(9, 2.3 * nvars), gridspec_kw={"width_ratios": [35, 1]}
    )
    for i in range(nvars):
        # Plot SURD
        information_flux[i + 1] = surd.plot(
            Rd_results[i + 1],
            Sy_results[i + 1],
            info_leak_results[i + 1],
            axs[i, :],
            nvars,
            threshold=-0.01,
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i + 1}}} / I \\left(Q_{i + 1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=12,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i + 1}}}}}{{H \\left(Q_{i + 1} \\right)}}$",
            pad=20,
        )
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=20,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )

    # Show the results
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels("")

    plt.tight_layout(w_pad=-8, h_pad=0)
    plot_save_path = f"./results/surd_results/{block}.png"
    plt.savefig(plot_save_path)
    plt.show()
