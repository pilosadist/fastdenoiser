from denoiser import DenoiseFile
import torchaudio
#
device = 'cuda'

d = DenoiseFile('w', order=5)

wav, sr = torchaudio.load('dataset\\noisy_test\\p232_155.wav')

out = d(wav.to(device))

torchaudio.save('examples\\p232_155.wav', wav.cpu().detach(), 48000)
torchaudio.save('examples\\p232_155_w_5.wav', out.cpu().detach(), 48000)
