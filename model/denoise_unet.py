import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(Down, self).__init__()
        self.out_ch = out_ch
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()

    def forward(self, x):
        c = self.relu(self.conv(x))
        w = torch.tensor([0.032830059166456364,
                          0.20153300372583,
                          0.4062497075602523,
                          0.5085794685992991,
                          0.4062497075602523,
                          0.20153300372583,
                          0.032830059166456364]).to(device)
        w = torch.ones(self.out_ch, 1, 7).to(device) * w
        d = F.conv1d(c, w, padding=3, groups=self.out_ch)
        d = self.pool(d)
        return c, d


class Neck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Neck, self).__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=in_ch, num_layers=1, batch_first=True)
        self.in_ch = in_ch
        self.out_ch = out_ch
        kernel_size = 3
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()
        self.device = device

    def forward(self, x):
        hidden = Variable(torch.zeros(1, x.size(0), self.in_ch).to(self.device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.in_ch).to(self.device))
        lstm_out, (hidden, c_0) = self.lstm(x.permute(0, 2, 1), (hidden, c_0))
        x = lstm_out.permute(0, 2, 1)
        return self.relu(self.conv(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, lin=True):
        super(Up, self).__init__()
        kernel_size = 7
        if lin:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, kernel_size=5, stride=2, padding=2)

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()
        self.conv_to_size = nn.Conv1d(in_ch - out_ch, out_ch, kernel_size, padding=kernel_size // 2)

    def forward(self, c, d):
        d = self.up(d)
        d = self.conv_to_size(d)
        x = c + d
        x = self.relu(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tan(x)
        return x


class UnetBlock(nn.Module):
    def __init__(self):
        super(UnetBlock, self).__init__()
        filters_inc = 24
        layers = 8
        kernels = [7, 7, 15, 15, 15, 15, 15, 15]
        self.downstream = nn.ModuleList([])
        self.downstream.append(Down(1, filters_inc, kernels.pop()))
        for layer in range(1, layers):
            self.downstream.append(
                Down(layer * filters_inc, (layer + 1) * filters_inc, kernels.pop())
            )
        self.neck = Neck(layers * filters_inc, (layers + 1) * filters_inc)
        self.upstream = nn.ModuleList([])
        for layer in range(layers - 1, -1, -1):
            self.upstream.append(
                Up((layer + 2 + layer + 1) * filters_inc, (layer + 1) * filters_inc)
            )
        self.out = OutConv(filters_inc + 1, 1)

    def forward(self, x):
        layer_out = x
        concatenation = []
        for layer in self.downstream:
            # print(layer_out.shape)
            c, layer_out = layer(layer_out)
            concatenation.append(c)
        # print(layer_out.shape)
        layer_out = self.neck(layer_out)
        # print(layer_out.shape)
        for layer in self.upstream:
            layer_out = layer(concatenation.pop(), layer_out)
        # print(layer_out.shape)
        x = torch.cat([x, layer_out], dim=1)
        x = self.out(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.u1 = UnetBlock()
        self.u2 = UnetBlock()

    def forward(self, x):
        return self.u1(x) - self.u2(x)
