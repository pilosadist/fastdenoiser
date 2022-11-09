from denoiser import DenoiseFile
import torchaudio

device = 'cpu'

d = DenoiseFile('u', order=2)

wav, sr = torchaudio.load('dataset\\noisy_test\\long.wav')

out = d(wav.to(device))

torchaudio.save('w_in.wav', wav.cpu().detach(), 44000)
torchaudio.save('w_out.wav', out.cpu().detach(), 44000)
