
import torch
from torch import nn

class three_linear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        #self.batch = nn.BatchNorm1d(out_features)
        #self.batch = nn.RMSNorm(out_features)

        self.cls_lin = nn.Linear(in_features, out_features)
        self.patch_linear = nn.Linear(in_features * 196, out_features)
        self.reg_linear = nn.Linear(in_features * 4, out_features)

    def forward(self, x):
        
        #4 register, 196 patch, 1 cls
        registers = x[:, 0:4, :]
        registers = registers.reshape(x.shape[0], -1)#Flatten

        patch = x[:, 4:-1, :]
        patch = patch.reshape(x.shape[0], -1)

        cls = x[:, -1 , :]


        registers = self.reg_linear(registers)
        patch = self.patch_linear(patch)
        cls = self.cls_lin(cls)
        
        
        xs = torch.stack((registers, patch, cls), axis = 1)
        
        #output shape is batch, 3, embedd
        
        #Either average the features, or learn some kind of combiner.
        x = torch.mean(xs, dim = 1)
#        x = self.batch(x)

        return x

