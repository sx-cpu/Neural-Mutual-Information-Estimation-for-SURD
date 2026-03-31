import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MineNet(nn.Module):
    def __init__(self, dim_x, dim_y, hidden=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)


# assume MineNet is defined above (unchanged)


class MINE:
    def __init__(
        self,
        dim_x,
        dim_y,
        lr=1e-4,
        device="cuda",
        ema_rate=0.01,
        lambda_reg=0.1,  # ### MOD: regularization strength for ReMINE (on g = log E[e^T])
        C_reg=0.0,  # ### MOD: anchor target C in d(g, C); paper often uses 0 or 1 (tuneable)
        window_size=500,  # ### MOD: window size K for micro-averaging / smoothing
    ):
        self.net = MineNet(dim_x, dim_y).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.device = device

        # EMA for optional smoothing (kept for training stability if desired)
        self.ma_et = None
        self.ma_rate = ema_rate

        # normalization params (fit externally before training)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # ReMINE hyperparams
        self.lambda_reg = float(
            lambda_reg
        )  # strength of regularizer on g = log E[exp(T)]
        self.C_reg = float(C_reg)  # anchor value for g
        self.window_size = int(window_size)

        # ### MOD: sliding buffers for micro-averaging (store recent batch statistics)
        # each entry is a tuple (t_batch_mean (float), et_batch_mean (float))
        self._stat_buffer = collections.deque(maxlen=self.window_size)

    #################################################################
    #  Normalization utilities
    #################################################################
    def fit_normalization(self, X, Y):
        # X,Y are expected as torch tensors (cpu is fine)
        Xc = X.detach().cpu()
        Yc = Y.detach().cpu()
        self.x_mean = Xc.mean(0, keepdim=True)
        self.x_std = Xc.std(0, keepdim=True) + 1e-8
        self.y_mean = Yc.mean(0, keepdim=True)
        self.y_std = Yc.std(0, keepdim=True) + 1e-8

    def normalize(self, X, Y):
        # move stats to device of inputs
        mx = self.x_mean.to(X.device)
        sx = self.x_std.to(X.device)
        my = self.y_mean.to(Y.device)
        sy = self.y_std.to(Y.device)
        return (X - mx) / sx, (Y - my) / sy

    @staticmethod
    def shuffle(Y):
        return Y[torch.randperm(len(Y))]

    #################################################################
    #  TRAIN: use batch-level g = log mean exp(T) for regularizer (ReMINE)
    #  and store (t_mean, et_mean) in sliding buffer for micro-averaging
    #################################################################

    def train(self, dataloader, epochs=20, clip_norm=1.0, print_every=1):
        """
        IMPORTANT: caller must call mine.fit_normalization(X, Y) before train().
        """
        self.net.train()

        mi_list, loss_list, ma_list = [], [], []
        for ep in range(1, epochs + 1):
            epoch_sum_mi = 0.0
            n_batches = 0

            for bx, by in dataloader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                # normalize batch (fit must have been done before calling train)
                bx, by = self.normalize(bx, by)

                self.opt.zero_grad()

                # joint score mean
                t = self.net(bx, by)  # shape (B,1)
                t_mean = t.mean()  # scalar tensor

                # marginal scores (batch-wise shuffle as negative sampling for training)
                by_marg = self.shuffle(by)
                t_marg = self.net(bx, by_marg)  # (B,1)

                # batch estimate of E[exp(T_marg)]
                # compute et_batch = mean(exp(t_marg))
                et_batch = torch.exp(t_marg).mean()

                # batch-level g (log E[exp(T)]) used in ReMINE regularizer
                # note: using batch estimate here as the local g; window averaging handled separately
                g_batch = torch.log(et_batch + 1e-12)

                # optional EMA over et (keeps backward compatibility/stability if you want)
                if self.ma_et is None:
                    self.ma_et = et_batch.detach()
                else:
                    self.ma_et = (
                        1 - self.ma_rate
                    ) * self.ma_et + self.ma_rate * et_batch.detach()

                # For training objective we use the EMA-smoothed denominator (stable) for the main loss term
                # but regularizer is applied on current batch g_batch to penalize drift of g.
                log_et_for_loss = torch.log(
                    self.ma_et + 1e-12
                )  # stable denominator for loss

                # ReMINE regularizer: d(g_batch, C) = (g_batch - C)^2  (L2)
                reg = (
                    self.lambda_reg * (g_batch - self.C_reg) ** 2
                )  ### MOD: regularize log-mean-exp

                # final loss (we minimize negative of ReMINE objective)
                # objective: t_mean - log_et_for_loss - lambda * (g_batch - C)^2
                loss = -(t_mean - log_et_for_loss) + reg
                # epoch_sum_loss += loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_norm)
                self.opt.step()

                # accumulate stats for printing + window buffer
                epoch_sum_mi += (t_mean - log_et_for_loss).item()
                n_batches += 1

                # ### MOD: push batch statistics into sliding buffer (for micro-averaging later)
                # store CPU floats to keep memory low
                self._stat_buffer.append(
                    (
                        float(t_mean.detach().cpu().item()),
                        float(et_batch.detach().cpu().item()),
                    )
                )

            if ep % print_every == 0:
                avg_mi = epoch_sum_mi / max(1, n_batches)
                # avg_loss = epoch_sum_loss / max(1, n_batches)
                mi_list.append(avg_mi)
                # loss_list.append(avg_loss.item())

                print(
                    f"Epoch {ep}/{epochs}, avg MI={avg_mi:.6f}, buffer_len={len(self._stat_buffer)}"
                )

        return mi_list

    #################################################################
    #  ESTIMATE: two modes
    #   - if use_window=True and buffer populated -> use micro-averaging over buffer
    #   - else -> compute fresh t_mean and K full-data permutations (exact-ish)
    #################################################################
    def estimate(self, X, Y, batch_size=65536, K=5, use_window=False):
        """
        If use_window=True and we have >=1 entries in the internal buffer, compute micro-averaged estimate:
           t_mean_window = mean over buffer of t_batch_means
           et_mean_window = mean over buffer of et_batch_means
           I_hat = t_mean_window - log(et_mean_window)
        Otherwise compute a full-data estimate using K global permutations (as before).
        """
        self.net.eval()
        X = X.to(self.device)
        Y = Y.to(self.device)

        # normalize inputs using stored stats
        Xn, Yn = self.normalize(X, Y)

        # if requested and buffer has info, use micro-averaging over sliding window
        if use_window and len(self._stat_buffer) > 0:
            # compute averages over stored batch stats
            t_vals, et_vals = zip(*list(self._stat_buffer))
            t_mean_window = float(np.mean(t_vals))
            et_mean_window = float(np.mean(et_vals))
            mi = t_mean_window - np.log(et_mean_window + 1e-12)
            return float(mi)

        # otherwise fall back to full-data K-permutation estimate
        dataset = TensorDataset(Xn, Yn)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # compute t_mean across full dataset
        t_chunks = []
        with torch.no_grad():
            for bx, by in loader:
                t_chunks.append(self.net(bx.to(self.device), by.to(self.device)).cpu())
        t_all = torch.cat(t_chunks, dim=0)
        t_mean = float(t_all.mean().item())

        # compute et via K global permutations (no clamp)
        N = len(Xn)
        Yn_cpu = Yn.cpu()
        et_ks = []
        for _ in range(K):
            perm = torch.randperm(N)
            Yp = Yn_cpu[perm]
            et_chunks = []
            idx = 0
            with torch.no_grad():
                for bx, _ in loader:
                    bs = bx.shape[0]
                    y_chunk = Yp[idx : idx + bs].to(self.device)
                    et_chunks.append(
                        torch.exp(self.net(bx.to(self.device), y_chunk)).cpu()
                    )
                    idx += bs
            et_ks.append(float(torch.cat(et_chunks).mean().item()))

        mean_et = float(np.mean(et_ks))
        mi = t_mean - np.log(mean_et + 1e-12)
        return float(mi)
