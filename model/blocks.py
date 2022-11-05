import torch
from torch import nn
import torch.nn.functional as F


class Conv1DFIR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):
        super(Conv1DFIR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        if kernel_size % 2:
            self.weight = nn.Parameter(
                0.01 * torch.randn(self.out_channels, self.in_channels, self.kernel_size // 2 + 1))
        else:
            self.weight = nn.Parameter(0.01 * torch.randn(self.out_channels, self.in_channels, self.kernel_size // 2))
        if bias:
            self.bias = nn.Parameter(0.01 * torch.randn(out_channels))
        else:
            self.bias = torch.zeros(out_channels)

    def forward(self, x):
        return F.conv1d(x,
                        torch.cat([self.weight, torch.flip(self.weight[:, :, :-1], [2])], dim=2),
                        bias=self.bias,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class Downsampling(nn.Module):
    def __init__(self, in_channels, kernel_size):
        self.conv = [Conv1DFIR(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               groups=in_channels) for _ in range(4)]
        self.alpha = nn.Parameter(torch.ones(in_channels, 1))

    def forward(self, x):
        odd = x[:, :, 1::2]
        even = x[:, :, ::2]
        odd = odd - self.conv[0](even)
        even = even + self.conv[1](odd)
        odd = odd - self.conv[2](even)
        even = even + self.conv[3](odd)
        odd = odd / self.alpha
        even = even * self.alpha
        return torch.cat([even, odd], dim=1)
