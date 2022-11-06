import torch
from torch import nn
import torch.nn.functional as F


class Conv1DFIR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = 0,
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
        w = 0.01 * torch.randn(self.out_channels, self.in_channels, self.kernel_size // 2 + 1)
        s = (2 * torch.sum(w[:, :, :-1], dim=2) + w[:, :, -1]) / kernel_size
        s = s.unsqueeze(dim=2)
        w = w - s

        self.weight = nn.Parameter(w, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(0.01 * torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        return F.conv1d(x,
                        torch.cat([self.weight, torch.flip(self.weight[:, :, :-1], [2])], dim=2),
                        bias=self.bias,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class Downsampling(nn.Module):
    def __init__(self, kernel_size):
        super(Downsampling, self).__init__()
        self.conv = [Conv1DFIR(in_channels=1,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               groups=1) for _ in range(4)]
        self.alpha = nn.Parameter(torch.Tensor([1]), requires_grad=True)

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


class Upsampling(nn.Module):
    def __init__(self, kernel_size):
        super(Upsampling, self).__init__()
        self.conv = [Conv1DFIR(in_channels=1,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               groups=1) for _ in range(4)]
        self.alpha = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.tanh = nn.Tanh()

    def forward(self, even, odd):
        odd = odd * self.alpha
        even = even / self.alpha
        even = even - self.conv[0](odd)
        odd = odd + self.conv[1](even)
        even = even - self.conv[2](odd)
        odd = odd + self.conv[3](even)
        mix = torch.cat([even, odd], dim=1).permute(0, 2, 1)
        mix = mix.reshape(mix.shape[0], 1, -1)
        return mix


class FIRFilter(nn.Module):
    def __init__(self):
        super(FIRFilter, self).__init__()
        self.conv = Conv1DFIR(in_channels=1,
                              out_channels=48,
                              kernel_size=15,
                              padding=7,
                              bias=False,
                              groups=2)

        self.regressor = nn.Sequential(nn.Linear(48, 96),
                                       nn.ReLU(),
                                       nn.Linear(96, 2),
                                       nn.Tanh())

    def forward(self, x):
        c = self.conv(x).permute(0, 2, 1)
        c = self.regressor(c).permute(0, 2, 1)
        return x + c