import torch
import torch.nn as nn
from dualbranchconvolution import DualConv
class SicsbdMOSFET(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,320),
            nn.BatchNorm1d(320),
            nn.ReLU()
        )
        self.reshape = (64,5)
        self.transposed_cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64,out_channels=32,kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.dualcnn = nn.Sequential(
            DualConv(32),
            DualConv(64),
            DualConv(128)
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(256, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
         nn.Flatten(),
         nn.Linear(32*5, 64),
         nn.BatchNorm1d(64),
         nn.ReLU(),
         nn.Linear(64, 32),
         nn.BatchNorm1d(32),
         nn.ReLU(),
         nn.Linear(32, 4),            
        )
    def forward(self, X):
        X = self.fc1(X)
        X = X.view(X.shape[0], *self.reshape)     
        X = self.transposed_cnn(X)
        X = self.dualcnn(X)
        X = self.cnn(X)
        X = self.fc2(X)
        return X
