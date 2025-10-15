import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNeX_PSD(nn.Module):
    def __init__(self, in_ch=64, n_classes=4, n_times=500):
        super().__init__()
        self.in_ch = in_ch
        self.n_times = n_times
        
        # Parallel PSD-like band filters
        self.band_delta = nn.Conv1d(in_ch, in_ch, kernel_size=64, dilation=2, padding='same', bias=False)
        self.band_theta = nn.Conv1d(in_ch, in_ch, kernel_size=32, dilation=2, padding='same', bias=False)
        self.band_alpha = nn.Conv1d(in_ch, in_ch, kernel_size=16, dilation=2, padding='same', bias=False)
        self.band_beta  = nn.Conv1d(in_ch, in_ch, kernel_size=8,  dilation=2, padding='same', bias=False)
        self.band_gamma = nn.Conv1d(in_ch, in_ch, kernel_size=4,  dilation=2, padding='same', bias=False)
        
        # Temporal convolution (input + 5 bands)
        self.temporal_conv = nn.Conv1d(in_ch * 6, 32, kernel_size=25, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ELU()

        # Depthwise spatial filter
        self.spatial_conv = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ELU()
        
        # Separable/residual stack (simple example)
        self.sep_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding='same', bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU()
        )
        
        # Pooling + classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # Parallel PSD filters
        delta = self.band_delta(x)
        theta = self.band_theta(x)
        alpha = self.band_alpha(x)
        beta  = self.band_beta(x)
        gamma = self.band_gamma(x)
        
        # Concatenate all with original input
        x = torch.cat([x, delta, theta, alpha, beta, gamma], dim=1)
        
        # Temporal convolution
        x = self.act1(self.bn1(self.temporal_conv(x)))
        
        # Spatial and separable conv blocks
        x = self.act2(self.bn2(self.spatial_conv(x)))
        x = self.sep_conv(x)
        
        # Global average pool
        x = self.pool(x).squeeze(-1)
        
        # Final classifier
        x = self.fc(x)
        return x