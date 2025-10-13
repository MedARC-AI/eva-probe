import torch
from torch import nn

class tied_double_linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        projection_dim: int = 64,
        n_patches: int = 196
    ):

        super().__init__()

        self.batch = nn.RMSNorm(in_features)
        self.projection = nn.Linear(in_features, projection_dim)

        self.cls_lin = nn.Linear(in_features, out_features)
        self.patch_linear = nn.Linear(projection_dim * n_patches, out_features)
        self.reg_linear = nn.Linear(in_features * 4, out_features)

    def forward(self, x):

        x = self.batch(x)

        B = x.shape[0]

        registers = x[:, 0:4, :]
        patch = x[:, 4:-1, :]
        cls = x[:, -1, :]

        projected_patches = self.projection(patch)
        projected_patches_flat = projected_patches.reshape(B, -1)

        registers_flat = registers.reshape(B, -1)

        registers_out = self.reg_linear(registers_flat)
        patch_out = self.patch_linear(projected_patches_flat)
        cls_out = self.cls_lin(cls)

        xs = torch.stack((registers_out, patch_out, cls_out), dim=1)

        x_out = torch.mean(xs, dim=1)

        return x_out
