from model import blocks
from torch import nn
import torch

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class LayerDown(nn.Module):
    def __init__(self):
        super(LayerDown, self).__init__()
        self.f = blocks.FIRFilter()
        self.down1 = blocks.Downsampling(7)
        self.down2 = blocks.Downsampling(7)


    def forward(self, x):
        x = self.f(x)
        return self.down1(x[:, 0:1, :]), self.down2(x[:, 1:2, :])


class LayerUp(nn.Module):
    def __init__(self):
        super(LayerUp, self).__init__()
        self.f = blocks.FIRFilter()
        self.up1 = blocks.Upsampling(7)
        self.up2 = blocks.Upsampling(7)

    def forward(self, skipx, upx):

        ch1 = self.up1(skipx[:, 0:1, :], upx[:, 0:1, :])
        ch2 = self.up2(skipx[:, 1:2, :], upx[:, 1:2, :])
        return self.f(torch.cat([ch1, ch2], dim=1))


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.downstream = [LayerDown() for _ in range(6)]
        self.upstream = [LayerUp() for _ in range(6)]
        self.neck = blocks.Neck(2)

    def forward(self, x):
        skips = []
        for i in self.downstream:
            x, s = i(x)
            skips.append(s)
        x = self.neck(x)
        for i in self.upstream:
            x = i(x, skips.pop())
        return x