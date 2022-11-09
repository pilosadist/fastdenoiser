import torch
import net_unet_deep_best
import torchaudio

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class DenoiseFile():
    def __init__(self, param_dict):
        self.model = net_unet_deep_best.Unet().to(device)
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
        wav_chunks = [torch.zeros(1, 512)] + [wav[:, i:i + 512] for i in range(0, wav_len, 512)]
        wav_chunks = [torch.cat((wav_chunks[i], wav_chunks[i + 1], wav_chunks[i + 2], wav_chunks[i + 3]), dim=1)
                      for i in range(1, len(wav_chunks) - 4, 2)]
        wav_chunks = [self.model(i.unsqueeze(0)).squeeze(dim=0) for i in
                      wav_chunks]
        wav_chunks = [i[:, 512:-512] for i in wav_chunks]
        out_wav = torch.cat(wav_chunks, dim=1)
        return out_wav

d = DenoiseFile('best_w.pt')

wav, sr = torchaudio.load('dataset\\noisy_test\\p257_004.wav')

out = d.denoise(d.denoise(d.denoise(wav.to(device))))

print(out.shape)

torchaudio.save('w_in.wav', wav.cpu().detach(), 44000)
torchaudio.save('w_out.wav', out.cpu().detach(), 44000)

#
