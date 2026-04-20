"""
Physics-Informed CNN (PI-CNN) U-Net Architecture
================================================
Модули 2 и 5: Архитектура нейронной сети и класс тренировки
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils_new import BSParams, BSLoss, CubicStretching

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
    def __init__(self, bs: BSParams, activation="gelu", features=[32, 64, 128, 256]):
        super().__init__()
        with torch.no_grad():
            self.bs = bs
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

        self.out_conv = nn.Conv2d(features[0], 1, kernel_size=1, bias=True)

        self._initialize_weights()
    def _initialize_weights(self):
        # Kaiming для conv слоёв
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                with torch.no_grad():
                    m.weight.data *= 0.9

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Специальная инициализация выходного слоя
        with torch.no_grad():
            nn.init.xavier_uniform_(self.out_conv.weight, gain=0.01)
            nn.init.constant_(self.out_conv.bias, 0.2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # S_norm = x[:, 0:1, :, :]
        # t_norm = x[:, 1:2, :, :]
        # V_net = self.out_conv(d1)
        #
        # payoff = self.bs.terminal_payoff(S_norm)
        # bc_Smax = self.bs.far_field_bc(S_norm, t_norm)
        #
        #
        # V = (t_norm * payoff +
        #      (1 - t_norm) * ((1 - S_norm) * 0.0 + S_norm * bc_Smax) +
        #      S_norm * (1 - S_norm) * (1 - t_norm) * V_net)
        nn_out = self.out_conv(d1)
        S_norm = x[:, 0:1, :, :]
        tau_norm = x[:, 1:2, :, :]
        # Жесткое условие: V(S, tau=0) = payoff(S) ровно
        payoff = torch.clamp(S_norm - self.bs.K / self.bs.S_max, min=0.0)
        V = tau_norm * nn_out + payoff
        return V



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
        u_vals = torch.linspace(0, 1, self.N_S, device=self.device)
        self.stretching = CubicStretching(
            B=self.bs.K, S_min=self.bs.S_min, S_max=self.bs.S_max,
            alpha=(self.bs.S_max - self.bs.S_min) / 10, device=self.device
        )
        S_stretched = self.stretching.transform(u_vals)
        S_norm_1d = (S_stretched - self.bs.S_min) / (self.bs.S_max - self.bs.S_min)

        #tau_norm идет от 0 (maturity) до 1 (now)
        tau_norm_1d = torch.linspace(0, 1, self.N_t, device=self.device)
        S_norm_2d, tau_norm_2d = torch.meshgrid(S_norm_1d, tau_norm_1d, indexing='ij')
        self.coords_input = torch.stack([S_norm_2d, tau_norm_2d], dim=0).unsqueeze(0)

        self.criterion = BSLoss(
            bs_params,
            S_norm_2d.unsqueeze(0).unsqueeze(0),
            tau_norm_2d.unsqueeze(0).unsqueeze(0),
            self.N_S, self.N_t, self.device,
            lambda_pde, lambda_bc, lambda_tc, lambda_violation
        )


        self.optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-8)

        self.history = {"loss": [], "pde": [], "boundary": [], "T": []}

    def train(self, verbose_freq=200):
        self.model.train()
        # Scale output to reasonable range
        assert self.criterion.S_grid_norm.max() <= 1.0 + 1e-6
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            V_pred = self.model(self.coords_input)
            loss, comps = self.criterion(V_pred)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            self.history["loss"].append(comps["total"])
            self.history["pde"].append(comps["pde"])
            self.history["boundary"].append(comps["boundary"])
            self.history["T"].append(comps["T"])

            if epoch % verbose_freq == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss {comps['total']:.2e} | PDE {comps['pde']:.2e} "
                      f"| boundary {comps['boundary']:.2e} | T {comps['T']:.2e} | LR {lr:.2e}")

    @torch.no_grad()
    def predict_on_grid(self):
        self.model.eval()
        V = self.model(self.coords_input).squeeze().cpu().numpy()  # [N_S, N_t], τ: 0→1
        V_denorm = V * self.bs.S_max

        # Физические координаты по S
        S_phys = self.stretching.transform(
            torch.linspace(0, 1, self.N_S, device=self.device)
        ).cpu().numpy()

        # τ → t = T - τ*T
        tau_phys = np.linspace(0, self.bs.T, self.N_t)  # 0 → T
        t_phys = self.bs.T - tau_phys  # T → 0

        S_grid, t_grid = np.meshgrid(S_phys, t_phys, indexing='ij')
        return S_grid, t_grid, V_denorm