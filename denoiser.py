import torch
from model import net
import torchaudio

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class DenoiseFile():
    def __init__(self, param_dict):
        self.model = net.Denoiser().to(device)
        self.model.train()
        self.model.load_state_dict(torch.load(param_dict))
        self.model.eval()

    def size_normalization(self, waveform):
        if waveform.shape[1] < 512:
            delta = 512 - waveform.shape[1]
            return torch.cat((waveform, torch.zeros(delta).unsqueeze(dim=0)), dim=1)
        return waveform

    def denoise(self, wav):
        wav_len = wav.shape[1]
        print(wav_len)
        wav_chunks = [torch.zeros(1, 512)] + [wav[:, i:i + 512] for i in range(0, wav_len, 512)]
        wav_chunks = [torch.cat((wav_chunks[i], wav_chunks[i + 1], wav_chunks[i + 2], wav_chunks[i + 3]), dim=1)
                      for i in range(1, len(wav_chunks) - 4, 2)]
        wav_chunks = [self.model(torch.cat([i.to(device), i.to(device)], dim=0).unsqueeze(0)).squeeze(dim=0) for i in
                      wav_chunks]
        wav_chunks = [i[0, 512:-512] for i in wav_chunks]
        out_wav = torch.cat(wav_chunks, dim=0)
        return out_wav.unsqueeze(0)


d = DenoiseFile('denoiser_299.pt')

wav, sr = torchaudio.load('dataset\\noisy_test\\p232_021.wav')

out = d.denoise(wav)

print(out.shape)

torchaudio.save('w_in.wav', wav.cpu().detach(), 44000)
torchaudio.save('w_out.wav', out.cpu().detach(), 44000)

#
#
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
# d = SpeachData(test=True, random_state=66)
# w_in, w_true = d[400]
# torchaudio.save('n_in.wav', (w_in - w_true).detach(), 16000)
#
# w_in = w_in.unsqueeze(dim=0)
# w_out = model(w_in.to(device))
# torchaudio.save('w_in.wav', w_in.cpu().squeeze(dim=0).detach(), 16000)
# torchaudio.save('w_true.wav', w_true.cpu().detach(), 16000)
# torchaudio.save('w_out.wav', w_out.cpu().squeeze(dim=0).detach(), 16000)
