

from torch import nn

class batch_linear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        #self.batch = nn.BatchNorm1d(in_features)
        self.batch = nn.RMSNorm(in_features)
        self.lin = nn.Linear(in_features, out_features)


    def forward(self, x):

        x = self.batch(x)
        x = self.lin(x)

        return x


