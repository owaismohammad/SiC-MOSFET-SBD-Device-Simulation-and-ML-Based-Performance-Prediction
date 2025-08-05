import torch
import torch.nn as nn
class DualConv(nn.Module):
    def __init__(self, in_channels:int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5,stride=1, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5,stride=1, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=5,stride=1, padding=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
    def forward(self, X):
        out1 = self.conv1(X)
        out2 = self.conv2(X)
        conc = torch.concat((out1,out2), dim =1)
        return conc        
