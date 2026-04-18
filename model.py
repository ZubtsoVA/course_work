"""
Physics-Informed CNN (PI-CNN) U-Net Architecture
================================================
Модули 2 и 5: Архитектура нейронной сети и класс тренировки
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import CubicStretching, BSParams, BSLoss

# --------------------------
# 2. U-Net Architecture
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}[activation]
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            act
        )
    def forward(self, x): return self.block(x)


class PiCNN_BlackScholes(nn.Module):
    def __init__(self, activation="gelu", features=[32, 64, 128, 256]):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(2, features[0], activation)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(features[0], features[1], activation)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(features[1], features[2], activation)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(features[2], features[3], activation)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(features[3] + features[2], features[2], activation)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(features[2] + features[1], features[1], activation)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(features[1] + features[0], features[0], activation)

        self.out_conv = nn.Conv2d(features[0], 1, kernel_size=1)
        # with torch.no_grad():
        #     self.out_conv.weight.data.normal_(0, 0.01)
        #     self.out_conv.bias.data.fill_(0.5)



    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        S_norm = x[:, 0:1, :, :]
        return (torch.tanh(out) * 0.5 + 0.5)*S_norm


# --------------------------
# 5. Trainer
# --------------------------
class Trainer:
    def __init__(self, model, bs_params, grid_shape=(128, 128),
                 lr=1e-3, epochs=2000, device=None,
                 lambda_pde=1.0, lambda_bc=10.0, lambda_tc=10.0, lambda_violation = 6.0):
        self.model = model
        self.bs = bs_params
        self.N_S, self.N_t = grid_shape
        self.epochs = epochs
        self.device = device or torch.device("cpu")

        # Physical grids
        S_vals = torch.linspace(0, 1, self.N_S, device=self.device)
        t_vals = torch.linspace(0, 1, self.N_t, device=self.device)
        S_grid, t_grid = torch.meshgrid(S_vals, t_vals, indexing='ij')

        # Normalized inputs for network
        S_norm = S_grid
        t_norm = t_grid
        self.coords_input = torch.stack([S_norm, t_norm], dim=0).unsqueeze(0)

        # Grid spacings
        h_S = 1.0 / (self.N_S - 1)
        h_t = 1.0 / (self.N_t - 1)

        self.stretching = CubicStretching(
            B=self.bs.K,
            S_min=self.bs.S_min,
            S_max=self.bs.S_max,
            alpha=1.0,  # Попробуйте от 1.0 до 5.0
            device=self.device
        )

        self.criterion = BSLoss(
            bs_params,
            S_grid.unsqueeze(0).unsqueeze(0),
            t_grid.unsqueeze(0).unsqueeze(0),
            self.N_S, self.N_t,
            self.device,
            self.stretching,
            lambda_pde, lambda_bc, lambda_tc, lambda_violation,

        )

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=epochs, eta_min=1e-6)
        self.history = {"loss": [], "pde": [], "Smax": [], "T": [], "S0": []}

    def train(self, verbose_freq=200):
        self.model.train()
        # Scale output to reasonable range
        assert self.criterion.S_grid_norm.max() <= 1.0 + 1e-6
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            V_pred = self.model(self.coords_input)
            loss, comps = self.criterion(V_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step(loss)

            # FIXED: Use correct keys that match comps dictionary
            self.history["loss"].append(comps["total"])
            self.history["pde"].append(comps["pde"])
            self.history["Smax"].append(comps["Smax"])
            self.history["S0"].append(comps["S0"])
            self.history["T"].append(comps["T"])

            if epoch % verbose_freq == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss {comps['total']:.2e} | PDE {comps['pde']:.2e} "
                      f"| Smax {comps['Smax']:.2e} | T {comps['T']:.2e} | S0 {comps['S0']:.2e} | LR {lr:.2e}")

    @torch.no_grad()
    def predict_on_grid(self):
        self.model.eval()
        V = self.model(self.coords_input).squeeze().cpu().numpy()
        V_denorm = V * self.bs.S_max
        S = np.linspace(self.bs.S_min, self.bs.S_max, self.N_S)
        t = np.linspace(0, self.bs.T, self.N_t)
        S_grid, t_grid = np.meshgrid(S, t, indexing='ij')
        return S_grid, t_grid, V_denorm
