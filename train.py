import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def plot_wav(file_path):
    audio_data, _ = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data)
    plt.title('Waveform of {}'.format(os.path.basename(file_path)))
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()

def add_noise(audio_data, noise_level=0.1):
    noise = np.random.normal(scale=noise_level, size=len(audio_data))
    noisy_audio = audio_data + noise
    return noisy_audio

def extract_features(file_path):
    audio_data, _ = librosa.load(file_path, sr=None)
    # mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=13)

    # return mfccs.ravel()
    return audio_data

class AudioDataset(Dataset):
    def __init__(self, file_paths, no_samples=50000):
        self.file_paths = file_paths
        self.no_samples = no_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio_data, _ = librosa.load(file_path, sr=None)
        if len(audio_data) < self.no_samples:
            audio_data = np.pad(audio_data, (0, self.no_samples - len(audio_data)), 'constant', constant_values=0)
        return audio_data[:self.no_samples]


file_paths = []
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)


train_paths, val_paths = train_test_split(file_paths[:100], test_size=0.2, random_state=42)
batch_size = 16
num_epochs = 30
train_dataset = AudioDataset(train_paths)
val_dataset = AudioDataset(val_paths)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

class DenoiserCNN(nn.Module):
    def __init__(self):
        super(DenoiserCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = DenoiserCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, audio_batch in enumerate(train_loader):
        outputs = model(audio_batch.unsqueeze(1))

        loss = criterion(outputs, audio_batch.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

val_loss = 0.0
with torch.no_grad():
    for audio_batch in val_loader:
        outputs = model(audio_batch.unsqueeze(1))

        val_loss += criterion(outputs, audio_batch.unsqueeze(1))

print(f'Validation Loss: {val_loss / len(val_loader)}')
