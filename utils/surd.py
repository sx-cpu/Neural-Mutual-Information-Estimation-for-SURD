from itertools import combinations as icmb
from typing import Dict

import matplotlib.colors as mcolors
import numpy as np


def compute_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def compute_info_leak(data, bins=50):
    """
    data: shape (N, d)  where:
        data[:,0] = target T
        data[:,1:] = agents A1, A2, ...
    """
    T = data[:, 0]
    A = data[:, 1:]

    # Histogram estimate for T
    p_T, _ = np.histogram(T, bins=bins, density=True)
    p_T = p_T / np.sum(p_T)

    # Histogram estimate for joint (T,A)
    # we flatten A to a tuple of bins
    joint_bins = [bins] * (1 + A.shape[1])
    p_TA, _ = np.histogramdd(data, bins=joint_bins, density=True)
    p_TA = p_TA / np.sum(p_TA)

    # p(A)
    p_A = np.sum(p_TA, axis=0)
    # p(T|A)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_T_given_A = p_TA / p_A[np.newaxis, :]

    # H(T)
    H_T = compute_entropy(p_T)

    # H(T|A)
    H_T_given_A = np.nansum(p_A * compute_entropy(p_T_given_A.reshape(bins, -1)))

    return H_T_given_A / H_T


def surd_global(MI: Dict[tuple, float], n_vars: int):
    """
    Full SURD decomposition using MI values from MLP estimator.
    Reproduces exactly the global SURD (Figure 13).

    Input:
      MI[(1,)] , MI[(2,)], ... , MI[(1,2)] , ...

    Output:
      I_R: redundant + unique
      I_S: synergy
      MI: original MI dictionary
    """

    # ---- Step 1: organize MI's by order ----
    T_sets = {}
    for comb, val in MI.items():
        k = len(comb)
        if k not in T_sets:
            T_sets[k] = []
        T_sets[k].append((comb, val))

    # ---- Step 2: sort each T̃M ascending ----
    for k in T_sets.keys():
        T_sets[k] = sorted(T_sets[k], key=lambda x: x[1])  # ascending MI

    # Prepare output
    I_R = {comb: 0.0 for comb in MI.keys()}
    I_S = {comb: 0.0 for comb in MI.keys() if len(comb) > 1}

    # ---- Step 3: Compute Redundant + Unique (only in T̃1) ----
    T1 = T_sets[1]  # list of (comb, val)
    n1 = len(T1)

    # convert to arrays
    I1_vals = np.array([v for (_, v) in T1])
    I1_combs = [c for (c, _) in T1]

    T1_indices = [next(iter(c)) for c in I1_combs]

    # I_0 = 0 for difference base
    prev = 0.0
    for i in range(n1):
        val = I1_vals[i]
        diff = val - prev
        if diff < 0:
            diff = max(diff, 0.0)

        if i < n1 - 1:
            for j in range(i, n1):
                suffix_indices = T1_indices[j:n1]  # list of ints
                suffix_key = tuple(sorted(suffix_indices))  # canonical ordering
                I_R[suffix_key] = diff
        else:
            # Unique: last element only
            last_idx = T1_indices[i]
            last_key = (last_idx,)
            I_R[last_key] = diff

        prev = val

    # ---- Step 4: Higher-order synergy (T̃2, T̃3,...) ----
    for M in range(2, n_vars + 1):
        TM = T_sets.get(M, [])
        if len(TM) == 0:
            continue

        IM_vals = np.array([v for (_, v) in TM])
        IM_combs = [c for (c, _) in TM]

        # max of previous order
        prev_max = max([v for (_, v) in T_sets[M - 1]])

        prev = 0.0
        for i in range(len(TM)):
            comb = IM_combs[i]
            val = IM_vals[i]

            if prev >= prev_max:
                diff = val - prev
            else:
                if val > prev_max:
                    diff = val - prev_max
                else:
                    diff = 0.0

            I_S[comb] += max(diff, 0.0)
            prev = val

    return I_R, I_S, MI


def nice_print(r_, s_, mi_, leak_):
    """Print the normalized redundancies, unique and synergy particles"""

    r_ = {key: value / max(mi_.values()) for key, value in r_.items()}
    s_ = {key: value / max(mi_.values()) for key, value in s_.items()}

    print("    Redundant (R):")
    for k_, v_ in r_.items():
        if len(k_) > 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Unique (U):")
    for k_, v_ in r_.items():
        if len(k_) == 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Synergystic (S):")
    for k_, v_ in s_.items():
        print(f"        {str(k_):12s}: {v_:5.4f}")

    print(f"    Information Leak: {leak_ * 100:5.2f}%")


def histogram_entropy(p):
    """Compute entropy from a probability vector p."""
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def compute_info_leak(Y, MI, bins=50):
    """Compute info leak = H(Y|X)/H(Y) = (H_Y - I)/H_Y"""

    # 1. Compute H(Y)
    y_bins = np.linspace(np.min(Y), np.max(Y), bins + 1)
    y_digitized = np.digitize(Y, y_bins) - 1
    y_digitized = np.clip(y_digitized, 0, bins - 1)

    Py = np.bincount(y_digitized, minlength=bins).astype(float)
    Py = Py / np.sum(Py)
    H_Y = histogram_entropy(Py)

    # 2. Compute I(Y;X_all)
    I_yx = max(MI.values())

    # 3. Info leak
    info_leak = (H_Y - I_yx) / H_Y

    return H_Y, I_yx, info_leak


def plot(I_R, I_S, info_leak, axs, nvars, threshold=0):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            I_R_keys.append(prefix + "".join(map(str, comb)))
            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")

    # Synergestic Contributions
    I_S_keys = [
        "S" + "".join(map(str, comb))
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]
    I_S_labels = [
        f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$"
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [
        I_R.get(tuple(map(int, key[1:])), 0)
        if "U" in key or "R" in key
        else I_S.get(tuple(map(int, key[1:])), 0)
        for key in label_keys
    ]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    if nvars == 2:
        axs[0].set_box_aspect(1 / 2.5)
    else:
        axs[0].set_box_aspect(1 / 4)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.0])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def plot_nlabels(I_R, I_S, info_leak, axs, nvars, nlabels=-1):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            I_R_keys.append(prefix + "".join(map(str, comb)))
            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")

    # Synergestic Contributions
    I_S_keys = [
        "S" + "".join(map(str, comb))
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]
    I_S_labels = [
        f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$"
        for r in range(2, nvars + 1)
        for comb in icmb(range(1, nvars + 1), r)
    ]

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [
        I_R.get(tuple(map(int, key[1:])), 0)
        if "U" in key or "R" in key
        else I_S.get(tuple(map(int, key[1:])), 0)
        for key in label_keys
    ]
    values /= sum(values)
    max_value = max(values)

    # Filtering based on threshold
    top_n_indices = np.argsort(values)[-nlabels:]

    # Filter both the values and labels arrays
    filtered_values = values[top_n_indices]
    filtered_labels = np.array(labels)[top_n_indices]
    original_order_indices = np.argsort(top_n_indices)
    filtered_values_in_original_order = filtered_values[original_order_indices]
    filtered_labels_in_original_order = filtered_labels[original_order_indices]

    # Convert filtered arrays back to lists if necessary
    values = filtered_values_in_original_order
    labels = filtered_labels_in_original_order.tolist()

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    axs[0].set_box_aspect(1 / 4)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.0])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))
