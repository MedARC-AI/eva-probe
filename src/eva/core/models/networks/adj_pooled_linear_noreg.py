import torch
from torch import nn
import math

class adj_pooled_linear_noreg(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pool_kernel_size = (7, 7),
        patch_grid_hw = (14, 14),
    ):
        super().__init__()

        self.in_features = in_features
        self.grid_h, self.grid_w = patch_grid_hw
        self.n_patches = self.grid_h * self.grid_w

        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError(f"patch_grid_hw must be positive, got {patch_grid_hw}.")

        k_h, k_w = pool_kernel_size
        if (self.grid_h % k_h) != 0 or (self.grid_w % k_w) != 0:
            raise ValueError(f"pool_kernel_size {pool_kernel_size} must tile patch_grid_hw {patch_grid_hw}.")

        self.pooled_h = self.grid_h // k_h
        self.pooled_w = self.grid_w // k_w
        pooled_n_patches = self.pooled_h * self.pooled_w

        self.batch = nn.RMSNorm(in_features)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)

        self.cls_lin = nn.Linear(in_features, out_features)
        self.patch_linear = nn.Linear(in_features * pooled_n_patches, out_features)

    def forward(self, x):

        x = self.batch(x)

        B = x.shape[0]

        patch = x[:, 4:-1, :]
        cls = x[:, -1, :]

        patch_grid = patch.reshape(B, self.grid_h, self.grid_w, self.in_features).permute(0, 3, 1, 2)
        pooled_patches_flat = self.pool(patch_grid).reshape(B, -1)

        patch_out = self.patch_linear(pooled_patches_flat)
        cls_out = self.cls_lin(cls)

        xs = torch.stack((patch_out, cls_out), dim=1)

        x_out = torch.mean(xs, dim=1)

        return x_out
