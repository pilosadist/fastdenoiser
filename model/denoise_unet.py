import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class SkipLSTM(nn.Module):
    def __init__(self, in_ch):
        super(SkipLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size= in_ch, num_layers=1, batch_first=True)
        self.in_ch = in_ch

    def forward(self, x):
        l = x.shape[2]
        seq = torch.cat([x, x], dim=2)
        lstm_out, (_, __) = self.lstm(seq.permute(0, 2, 1))
        lstm_out = lstm_out.permute(0, 2, 1)
        return lstm_out[:, :, l:] + x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, lstm=False):
        super(Down, self).__init__()
        self.out_ch = out_ch
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.convd = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.convc = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()
        self.lstmout = SkipLSTM(out_ch)
        self.lstm = lstm

    def forward(self, x):
        d = self.relu(self.convd(x))
        c = self.relu(self.convc(x))
        w = torch.tensor([-0.053253938645799054,
                          -0.019515044637716923,
                          0.18592030013491284,
                          0.4886808330764983,
                          0.6373420888528188,
                          0.4886808330764983,
                          0.18592030013491284,
                          -0.019515044637716923,
                          -0.053253938645799054]).to(device)
        w = torch.ones(self.out_ch, 1, 9).to(device) * w
        d = F.conv1d(d, w, padding=4, groups=self.out_ch)
        d = self.pool(d)
        if self.lstm:
            return self.lstmout(c), d
        else:
            return c, d


class Neck(nn.Module):
    def __init__(self, in_ch):
        super(Neck, self).__init__()
        self.lstm = nn.LSTM(input_size=in_ch * 2, hidden_size=in_ch * 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.in_ch = in_ch
        kernel_size = 3  # 7
        self.conv_in = nn.Conv1d(in_ch, in_ch * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_out = nn.Conv1d(in_ch * 4, in_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv_in(x))
        hidden = Variable(torch.zeros(2, x.size(0), self.in_ch * 2).to(device))
        c_0 = Variable(torch.zeros(2, x.size(0), self.in_ch * 2).to(device))
        lstm_out, (hidden, c_0) = self.lstm(x.permute(0, 2, 1), (hidden, c_0))
        x = lstm_out.permute(0, 2, 1)
        x = self.relu(self.conv_out(x))
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, lin=True):
        super(Up, self).__init__()
        kernel_size = 5  # 7
        if lin:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_ch // 2, in_ch // 2, kernel_size=5, stride=2, padding=2)

        self.conv = nn.Conv1d(in_ch * 2, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.Tanh()

    def forward(self, c, d):
        d = self.up(d)
        x = torch.cat((c, d), dim=1)
        x = self.relu(self.conv(x))
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_ch, 1, 1)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tan(x)
        return x


class UnetBlock(nn.Module):
    def __init__(self):
        super(UnetBlock, self).__init__()
        filters_inc = 24
        layers = 9
        kernels = [3, 5, 7, 15, 15, 15, 15, 15, 15]
        self.input = nn.Conv1d(1, filters_inc, 15, padding=7)
        self.downstream = nn.ModuleList([])
        for n, layer in enumerate(range(1, layers)):
            if n > 3:
                self.downstream.append(
                    Down(layer * filters_inc, (layer + 1) * filters_inc, kernels.pop(), lstm=True))
            else:
                self.downstream.append(
                    Down(layer * filters_inc, (layer + 1) * filters_inc, kernels.pop(), lstm=False))

        self.neck = Neck(layers * filters_inc)
        self.upstream = nn.ModuleList([])
        for layer in range(layers - 1, 0, -1):
            self.upstream.append(
                Up((layer + 1) * filters_inc, layer * filters_inc)
            )
        self.out = OutConv(filters_inc)

    def forward(self, x):
        layer_out = self.input(x)
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
        # print(x.shape)
        x = self.out(layer_out)
        return x


class UnetW(nn.Module):
    def __init__(self):
        super(UnetW, self).__init__()
        self.u1 = UnetBlock()
        self.u2 = UnetBlock()
        self.a = nn.parameter.Parameter(torch.tensor([0.33]).to(device))
        self.b = nn.parameter.Parameter(torch.tensor([0.33]).to(device))
        self.c = nn.parameter.Parameter(torch.tensor([0.33]).to(device))
        self.d = nn.parameter.Parameter(torch.tensor(0.5).to(device))
        self.f = nn.parameter.Parameter(torch.tensor(0.5).to(device))

    def forward(self, x):
        u1 = self.u1(x)
        u2 = self.u2(self.d * u1 + self.f * x)
        o = self.a * x + self.b * u1 + self.c * u2
        return o



