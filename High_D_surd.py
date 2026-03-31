"""
MINE implementation matching the paper's Eq.(6) and Fig.2 design idea.

Requirements:
  pip install torch torchvision

Usage:
  - Prepare dataset that yields pairs (q_plus_O, q_i), each tensor shape (C, H, W).
  - Instantiate MINEEstimator, then call train_mine(...) passing DataLoader of pairs.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MINEEstimator(nn.Module):
    """
    g_theta: maps stacked [Q_plus_O, Q_i] (channels = Nc) with spatial size ~64x32 to scalar MI score.
    Architecture follows paper Fig.2 idea: conv blocks downsampling + dense layers -> scalar.
    Sizes chosen to be flexible; adjust channels / widths as needed.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # conv encoder: mimic 64x32xNc -> down to small spatial features
        self.enc = nn.Sequential(
            ConvBlock(
                in_channels, 8, kernel=3, stride=1, padding=1
            ),  # keep spatial, increase channels
            ConvBlock(8, 16, kernel=3, stride=2, padding=1),  # downsample
            ConvBlock(16, 32, kernel=3, stride=2, padding=1),
            ConvBlock(32, 64, kernel=3, stride=2, padding=1),
        )
        # compute flattened size dynamically in forward (so code is flexible)
        # fully connected head (paper: 2048 -> 512 -> 32 -> scalar); we approximate similarly
        self.fc1 = nn.Linear(
            64 * 8 * 4, 512
        )  # <--- assumes input HxW around 64x32 -> after 3x stride2 downsample -> 8x4
        self.fc2 = nn.Linear(512, 32)
        self.fc_out = nn.Linear(32, 1)
        self.act = nn.ReLU()

        # small initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: tensor of shape (B, Nc, H, W) where Nc = channels of Q_plus_O + channels of Q_i (stacked)
        returns: (B, 1) scalar score g_theta for each pair
        """
        h = self.enc(x)  # (B, C', H', W')
        B = h.shape[0]
        h = h.view(B, -1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        out = self.fc_out(h)  # (B,1)
        return out


def mine_loss_from_scores(g_joint: torch.Tensor, g_marginal: torch.Tensor):
    """
    Compute DV-based MINE estimate:
      I_hat = mean(g_joint) - log(mean(exp(g_marginal)))
    We will minimize negative I_hat -> loss = -I_hat

    For stability compute log(mean(exp(...))) as logsumexp - log(n)
    """
    # g_joint, g_marginal: shape (B,1) or (B,)
    g_joint = g_joint.view(-1)
    g_marginal = g_marginal.view(-1)
    mean_joint = torch.mean(g_joint)

    # stable log mean exp
    # log(mean(exp(x))) = logsumexp(x) - log(n)
    lse = torch.logsumexp(g_marginal, dim=0)
    log_mean_exp = lse - math.log(g_marginal.numel())

    i_hat = mean_joint - log_mean_exp
    loss = -i_hat  # we maximize I_hat -> minimize negative
    return loss, i_hat.item()


def compute_marginal_by_shuffle(q_plus_O: torch.Tensor, q_i: torch.Tensor, device=None):
    """
    Given batch tensors:
      q_plus_O: (B, C1, H, W)
      q_i:      (B, C2, H, W)
    Return shuffled_q_plus_O (B, C1, H, W) where we permute along batch dim to break joint dependence.
    This follows the paper's marginal-sample construction.
    """
    B = q_plus_O.shape[0]
    idx = torch.randperm(B, device=(q_plus_O.device if device is None else device))
    return q_plus_O[idx]


def train_mine(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    log_every: int = 50,
):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_i = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # assume dataset returns tuple (q_plus_O, q_i) each as (B, C, H, W)
            q_plus_O, q_i = batch
            q_plus_O = q_plus_O.to(device)
            q_i = q_i.to(device)

            # Construct inputs for joint pairs: stack along channel dim per paper
            joint_input = torch.cat([q_plus_O, q_i], dim=1)  # (B, Nc, H, W)

            # Marginal: shuffle q_plus_O across batch
            q_plus_O_shuffled = compute_marginal_by_shuffle(
                q_plus_O, q_i, device=device
            )
            marginal_input = torch.cat([q_plus_O_shuffled, q_i], dim=1)

            # Forward
            g_joint = model(joint_input)  # (B,1)
            g_marginal = model(marginal_input)

            # Loss: negative DV bound
            loss, i_hat = mine_loss_from_scores(g_joint, g_marginal)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_i += i_hat

            if (batch_idx + 1) % log_every == 0:
                avg_loss = running_loss / log_every
                avg_i = running_i / log_every
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx + 1}]  loss={avg_loss:.4f}  I_hat={avg_i:.4f}"
                )
                running_loss = 0.0
                running_i = 0.0

        # epoch end: optionally print last avg
        print(f"Epoch {epoch + 1} finished.")

    print("Training finished.")


# -------------------------
# Example dataset & usage
# -------------------------
class ExamplePairDataset(Dataset):
    """
    Toy dataset producing pairs (q_plus_O, q_i).
    Replace with your real data loader that returns the two fields as tensors.
    Shapes used here: C=1 per field, H=64, W=32 as in paper diagrams.
    """

    def __init__(self, n_samples=2000, H=64, W=32):
        super().__init__()
        self.n = n_samples
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # synthetic correlated pair for demo: q_plus_O = f(q_i) + noise
        # q_i: (1,H,W) uniform noise
        q_i = torch.randn(1, self.H, self.W)
        q_plus_O = torch.sin(q_i * 2.0) + 0.001 * torch.randn_like(q_i)
        return q_plus_O.float(), q_i.float()


def example_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExamplePairDataset(n_samples=4000, H=64, W=32)
    dl = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)

    # Nc = channels of q_plus_O + q_i = 1 + 1 = 2
    model = MINEEstimator(in_channels=2)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_mine(model, dl, optim, device, epochs=50, log_every=20)

    # After training you can compute dataset-level I_hat by averaging batches:
    model.eval()
    with torch.no_grad():
        i_vals = []
        for q_plus_O, q_i in dl:
            q_plus_O = q_plus_O.to(device)
            q_i = q_i.to(device)
            joint = torch.cat([q_plus_O, q_i], dim=1)
            shuffled = torch.cat(
                [compute_marginal_by_shuffle(q_plus_O, q_i, device=device), q_i], dim=1
            )
            g_j = model(joint)
            g_m = model(shuffled)
            _, i_hat = mine_loss_from_scores(g_j, g_m)
            i_vals.append(i_hat)
        print("Estimated MI (avg over batches):", sum(i_vals) / len(i_vals))


if __name__ == "__main__":
    example_run()
