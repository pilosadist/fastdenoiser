import os
import torch
import torchaudio
import random


class SpeachData(torch.utils.data.IterableDataset):
    def __init__(self, path='dataset', test=False, random_state=0):
        self.clean_test_files = os.listdir(os.path.join(path, 'clean_test'))
        self.clean_train_files = os.listdir(os.path.join(path, 'clean_train'))
        self.noise_test_files = os.listdir(os.path.join(path, 'noisy_test'))
        self.noise_train_files = os.listdir(os.path.join(path, 'noisy_train'))
        self.train = [file_name for file_name in self.clean_train_files if file_name in self.noise_train_files]
        random.Random(random_state).shuffle(self.train)
        self.test = [file_name for file_name in self.clean_test_files if file_name in self.noise_test_files]
        random.Random(random_state).shuffle(self.test)
        self.test_mode = test
        self.path = path

    def size_normalization(self, waveform):
        if waveform.shape[1] < 2048:
            delta = 2048 - waveform.shape[1]
            return torch.cat((waveform, torch.zeros(delta).unsqueeze(dim=0)), dim=1)
        return waveform

    def __iter__(self):
        self.position = -1
        self.f = ''
        return self

    def __next__(self):
        if self.test_mode:
            if self.position < 0:
                try:
                    self.f = self.test.pop()
                    self.position = 0
                except:
                    raise StopIteration
            signal_path = os.path.join(self.path, 'clean_test')
            noise_path = os.path.join(self.path, 'noisy_test')
            signal_waveform, signal_sr = torchaudio.load(os.path.join(signal_path, self.f))
            noise_waveform, noise_sr = torchaudio.load(os.path.join(noise_path, self.f))
            if self.position < signal_waveform.shape[1]:
                if (self.position + 2048) < signal_waveform.shape[1]:
                    clean = signal_waveform[:, self.position:self.position + 2048]
                    noise = noise_waveform[:, self.position:self.position + 2048]
                    self.position += 2048
                else:
                    clean = self.size_normalization(noise_waveform[:, self.position:-1])
                    noise = self.size_normalization(noise_waveform[:, self.position:-1])
                    self.position = -1
                return noise, clean
        else:
            if self.position < 0:
                try:
                    self.f = self.train.pop()
                    self.position = 0
                except:
                    raise StopIteration
            signal_path = os.path.join(self.path, 'clean_train')
            noise_path = os.path.join(self.path, 'noisy_train')
            signal_waveform, signal_sr = torchaudio.load(os.path.join(signal_path, self.f))
            noise_waveform, noise_sr = torchaudio.load(os.path.join(noise_path, self.f))
            if self.position < signal_waveform.shape[1]:
                if (self.position + 2048) < signal_waveform.shape[1]:
                    clean = signal_waveform[:, self.position:self.position + 2048]
                    noise = noise_waveform[:, self.position:self.position + 2048]
                    self.position += 2048
                else:
                    clean = self.size_normalization(noise_waveform[:, self.position:-1])
                    noise = self.size_normalization(noise_waveform[:, self.position:-1])
                    self.position = -1
                return noise, clean
