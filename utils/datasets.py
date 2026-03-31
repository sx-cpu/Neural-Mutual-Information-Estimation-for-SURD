import numpy as np
import torch


# ===============================
# 2. Build lagged dataset
# ===============================
def build_lagged_dataset(q1, q2, q3, inputs, lag):
    mapping = {1: q1, 2: q2, 3: q3}
    X = np.vstack([mapping[i] for i in inputs]).T
    Y = q3.copy()
    X = X[:-lag]
    Y = Y[lag:]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(
        Y.reshape(-1, 1), dtype=torch.float32
    )


# ----------------------------------------
# 3. All combinations
# ----------------------------------------
def all_subsets(indices, max_k=None):
    """
    indices = [1,2,3]
    returns: (1,), (2,), (3,), (1,2), (1,3), (2,3), ...
    """
    import itertools

    subsets = []
    for k in range(1, len(indices) + 1):
        if (max_k is not None) and (k > max_k):
            continue
        for s in itertools.combinations(indices, k):
            subsets.append(s)
    return subsets
