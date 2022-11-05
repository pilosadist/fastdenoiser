import torch
from torch import nn

class BandDenoise(nn.Module):
    def __init__(self, out_channel, hidden_size):
        super(BandDenoise, self).__init__()

