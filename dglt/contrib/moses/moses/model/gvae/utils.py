import torch.nn as nn

class Flatten(nn.Module):
    """Flatten tensor except for the first dimension."""

    def forward(self, x):
        return x.view(x.size()[0], -1)
