import torch
from model import denoise_unet
from pympler.asizeof import asizeof

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class DenoiseFile:
    def __init__(self, model='w', order=1):
        if model == 'w':
            self.model = denoise_unet.Unet().to(device)
            self.model.train()
            self.model.load_state_dict(torch.load('weights\\best_w.pt', map_location=torch.device(device)))
        if model == 'u':
            self.model = denoise_unet.UnetBlock().to(device)
            self.model.train()
            self.model.load_state_dict(torch.load('weights\\best.pt', map_location=torch.device(device)))
        self.model.eval()
        self.model = [self.model for i in range(order)]

    def composition_model(self, x):
        with torch.inference_mode():
            for f in self.model:
                x = f(x)
        return x

    def size_normalization(self, waveform):
        if waveform.shape[1] < 512:
            delta = 512 - waveform.shape[1]
            return torch.cat((waveform, torch.zeros(delta).unsqueeze(dim=0)), dim=1)
        return waveform

    def denoiser(self, wav):
        wav_len = wav.shape[1] // 512 - 4
        chunk = torch.cat([torch.zeros(1, 512), wav[:, :512 * 3]], dim=1)
        chunk = self.composition_model(chunk.unsqueeze(0)).squeeze(dim=0)[:, 512:-512]
        for position in range(2, wav_len, 2):
            u = self.composition_model(wav[:, 512 * position:512 * (position + 4)].unsqueeze(0)).squeeze(dim=0)
            chunk = torch.cat([chunk, u[:, 512:-512]], dim=1)
            if position % 100 == 0:
                print(position, chunk.element_size() * chunk.nelement(), u.element_size())
        return chunk

    def __call__(self, wav):
        return self.denoiser(wav)
