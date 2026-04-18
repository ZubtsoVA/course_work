"""
Physics-Informed CNN (PI-CNN) Utilities
=======================================
Модули 0, 3, 4: Cubic Stretching, конечно-разностные операторы, функция потерь, параметры BS
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple


# --------------------------
# 0. Cubic Stretching
# --------------------------
class CubicStretching:
    """
    Реализация Eq. (2) из статьи 2210.02541v4.
    Преобразует равномерную координату u в физическую S с сгущением вокруг B.
    """

    def __init__(self, B: float, S_min: float, S_max: float, alpha: float, device: torch.device):
        self.B = B
        self.S_min = S_min
        self.S_max = S_max
        self.alpha = alpha
        self.chi = 6.0
        self.device = device

        # Решение кубических уравнений для c1 и c2
        # 1/6 * c^3 + c + Q = 0
        self.c1 = self._solve_cubic((B - S_min) / alpha)
        self.c2 = self._solve_cubic((B - S_max) / alpha)

    def _solve_cubic(self, Q: float) -> float:
        """Решает 1/6 c^3 + c + Q = 0 методом Ньютона"""
        c = -Q  # Начальное приближение
        for _ in range(5):  # 5 итераций достаточно для машинной точности
            f = (1 / 6) * c ** 3 + c + Q
            df = (1 / 2) * c ** 2 + 1
            c = c - f / df
        return c

    def compute_metrics(self, u: torch.Tensor):
        """
        Возвращает S(u), dS/du, d2S/du2 для заданной сетки u.
        Используется для преобразования производных CNN.
        """
        # Линейная интерполяция коэффициентов: L(u) = c2*u + c1*(1-u)
        L = self.c2 * u + self.c1 * (1 - u)
        dL_du = self.c2 - self.c1  # Константа

        # S(u)
        term = (1 / self.chi) * L ** 3 + L
        S = self.B + self.alpha * term

        # dS/du = alpha * dL_du * (0.5*L^2 + 1)
        dS_du = self.alpha * dL_du * (0.5 * L ** 2 + 1.0)

        # d2S/du2 = alpha * dL_du^2 * L
        d2S_du2 = self.alpha * (dL_du ** 2) * L

        return S, dS_du, d2S_du2


# --------------------------
# 1. Black-Scholes Parameters
# --------------------------
@dataclass
class BSParams:
    r: float = 0.05
    sigma: float = 0.20
    K: float = 100.0
    T: float = 1.0
    S_max: float = 300.0
    option: str = "call"
    S_min: float = 0.0  # Changed from 1e-3 to 0 for proper boundary

    def terminal_payoff(self, S: torch.Tensor) -> torch.Tensor:
        if self.option == "call":
            return torch.nn.functional.softplus(S - self.K / self.S_max, beta=50)
        else:
            return torch.nn.functional.softplus(self.K / self.S_max - S, beta=50)

    def far_field_bc(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Boundary condition at S = S_max."""
        if self.option == "call":
            tau = 1 - t
            return 1 - self.K * torch.exp(-self.r * tau) / self.S_max
        else:
            return torch.zeros_like(S)

    def analytical_price(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Closed-form Black-Scholes formula."""
        from scipy.stats import norm
        tau = self.T - t
        tau = np.maximum(tau, 1e-8)
        # Handle S=0 case
        S_safe = np.maximum(S, 1e-8)
        d1 = (np.log(S_safe / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        if self.option == "call":
            return S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


# --------------------------
# 3. Finite-Difference Operators
# --------------------------
class FDOperators:
    def __init__(self, device: torch.device, dS_norm: float, dt_norm: float):
        self.device = device
        self.dS_norm = dS_norm
        self.dt_norm = dt_norm

        # First derivative ∂/∂S_norm (central difference)
        kernel_dS = torch.zeros(1, 1, 3, 1, device=device)
        kernel_dS[0, 0, 0, 0] = -1.0
        kernel_dS[0, 0, 2, 0] = 1.0
        kernel_dS = kernel_dS / 2 / dS_norm  # Scale by grid spacing
        self.conv_dS = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0), padding_mode='replicate', bias=False, )
        self.conv_dS.weight = nn.Parameter(kernel_dS, requires_grad=False)

        # Second derivative ∂²/∂S_norm²
        kernel_d2S = torch.zeros(1, 1, 3, 1, device=device)
        kernel_d2S[0, 0, 0, 0] = 1.0
        kernel_d2S[0, 0, 1, 0] = -2.0
        kernel_d2S[0, 0, 2, 0] = 1.0
        kernel_d2S /= dS_norm ** 2
        self.conv_d2S = nn.Conv2d(1, 1, kernel_size=(3, 1), padding=(1, 0), padding_mode='replicate', bias=False)
        self.conv_d2S.weight = nn.Parameter(kernel_d2S, requires_grad=False)

        # Time derivative ∂/∂t_norm
        kernel_dt = torch.zeros(1, 1, 1, 3, device=device)
        kernel_dt[0, 0, 0, 0] = -1.0
        kernel_dt[0, 0, 0, 2] = 1.0
        kernel_dt = kernel_dt / 2 / dt_norm
        self.conv_dt = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0,1), padding_mode='replicate', bias=False, )
        self.conv_dt.weight = nn.Parameter(kernel_dt, requires_grad=False)

    def dS(self, u): return self.conv_dS(u)
    def d2S(self, u): return self.conv_d2S(u)
    def dt(self, u): return self.conv_dt(u)


# --------------------------
# 4. Loss Function
# --------------------------

class BSLoss(nn.Module):
    def __init__(self, bs: BSParams, S_grid: torch.Tensor, t_grid: torch.Tensor,
                 N_S: int, N_t: int, device: torch.device,
                 stretching: CubicStretching,lambda_pde=1.0, lambda_bc=10.0, lambda_tc=10.0, lambda_violation = 6.0):
        super().__init__()
        self.bs = bs
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_tc = lambda_tc
        self.lambda_violation = lambda_violation
        self.device = device
        self.V_scale = self.bs.S_max
        self.S_phys, self.S_u, self.S_uu = stretching.compute_metrics(S_grid)
        self.t_phys = t_grid * bs.T
        self.N_S = N_S
        self.N_t = N_t
        self.huber_delta = 0.01
        # Physical grid spacings
        self.dS_phys = self.bs.S_max / (N_S - 1)
        self.dt_phys = self.bs.T / (N_t - 1)

        self.S_grid_norm = S_grid
        self.t_grid_norm = t_grid
        self.fd = FDOperators(device, dS_norm=1.0/(N_S-1), dt_norm=1.0/(N_t-1))

        # Physical coordinates
        self.S_phys_for_mask = self.S_grid_norm * self.bs.S_max
        self.t_phys = self.t_grid_norm * self.bs.T
        self.nonzero_mask = (self.S_phys_for_mask > self.bs.K).float()


        N_S_grid, N_t_grid = S_grid.shape[2], t_grid.shape[3]

        # Create masks
        eps = 1e-2
        S_idx = torch.arange(N_S_grid, device=device)
        t_idx = torch.arange(N_t_grid, device=device)
        S_indices = torch.linspace(0, 1, N_S_grid, device=device)
        t_indices = torch.linspace(0, 1, N_t_grid, device=device)
        S_mesh, _ = torch.meshgrid(S_indices, t_indices)
        self.mask_itm = (self.S_phys > self.bs.K).float() # для штрафа нулевого решения
        self.mask_S0 = torch.zeros(1, 1, N_S_grid, N_t_grid, device=device)
        self.mask_Smax = torch.zeros(1, 1, N_S_grid, N_t_grid, device=device)
        self.mask_T = torch.zeros(1, 1, N_S, N_t, device=device)
        self.mask_itm = torch.zeros(1, 1, N_S_grid, N_t_grid, device=device)
        self.mask_S0.index_fill_(dim=2, index=torch.tensor([0], device=device), value=1.0)
        self.mask_Smax.index_fill_(dim=2, index=torch.tensor([N_S - 1], device=device), value=1.0)
        self.mask_T.index_fill_(dim=3, index=torch.tensor([N_t - 1], device=device), value=1.0)
        self.mask_itm = (S_mesh > self.bs.K / self.bs.S_max).float()

        # Маска для t=0 (начальный момент) и больших S
        self.mask_t0 = torch.zeros(1, 1, N_S, N_t, device=device)
        self.mask_t0.index_fill_(dim=3, index=torch.tensor([0], device=device), value=1.0)

        # Маска для больших S при t=0 (S > 0.7*S_max)
        self.mask_t0_largeS = self.mask_t0 * (self.S_grid_norm > 0.95).float()
        self.n_t0_largeS = self.mask_t0_largeS.sum()
        #Убираем пересечения
        # self.mask_T.index_fill_(dim=2, index=torch.tensor([N_S - 1], device=device), value=0.0)
        # self.mask_T.index_fill_(dim=2, index=torch.tensor([0], device=device), value=0.0)

        #self.mask_interior = ((S_grid >= eps) & (S_grid <= 1 - eps) & (t_grid <= 1 - eps) & ((S_grid - self.bs.K / self.bs.S_max > eps) | \
        #                      self.bs.K / self.bs.S_max - S_grid > eps )).float()
        self.mask_interior = torch.relu(torch.ones_like(S_grid) -self.mask_S0 - self.mask_Smax - self.mask_T - self.mask_t0)

        self.n_int = self.mask_interior.sum()
        self.n_S0 = self.mask_S0.sum()
        self.n_Smax = torch.sum(self.mask_Smax)
        self.n_T = self.mask_T.sum()
        self.n_itm = self.mask_itm.sum()

    def forward(self, V_norm: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Convert to physical value
        S_norm = self.S_grid_norm
        V_u = self.fd.dS(V_norm)
        V_uu = self.fd.d2S(V_norm)
        V_t = self.fd.dt(V_norm)
        S_u = self.S_u
        S_uu = self.S_uu
        # 2. Применяем правило цепочки для перехода к физическим производным
        # V_S = V_u / S_u
        V_S = V_u / S_u

        # V_SS = (V_uu * S_u - V_u * S_uu) / (S_u)^3
        V_SS = (V_uu * S_u - V_u * S_uu) / (S_u ** 3)

        # 3. Формируем PDE residual в физических переменных
        # ∂V/∂t = (1/T) * V_t
        V_t_phys = V_t / self.bs.T

        residual = V_t_phys + \
                   0.5 * self.bs.sigma ** 2 * S_norm ** 2 * V_SS + \
                   self.bs.r * S_norm * V_S - \
                   self.bs.r * V_norm

        # Добавить небольшую вязкость для стабилизации
        # viscosity = 0.001 * self.dS_phys * V_SS.abs().mean()
        # Black-Scholes residual real
        # residual = dV_dt + \
        #            0.5 * self.bs.sigma ** 2 * S_phys ** 2 * d2V_dS2 + \
        #            self.bs.r * S_phys * dV_dS - self.bs.r * V
        # PDE loss (interior only)

        pde_loss = ((residual * self.mask_interior) ** 2).sum() / self.n_int # + viscosity

        # Boundary conditions

        loss_S0 = ((V_norm * self.mask_S0) ** 2).sum() / self.n_S0
        V_target_Smax_norm = self.bs.far_field_bc(S_norm, self.t_grid_norm)

        loss_Smax = (((V_norm - V_target_Smax_norm) * self.mask_Smax) ** 2).sum() / self.n_Smax

        V_target_T_norm = self.bs.terminal_payoff(S_norm)

        diff_T = (V_norm - V_target_T_norm) * self.mask_T
        loss_T = torch.where(
            torch.abs(diff_T) < self.huber_delta,
            0.5 * diff_T ** 2,
            self.huber_delta * (torch.abs(diff_T) - 0.5 * self.huber_delta)
        ).sum() / self.n_T


        total = self.lambda_pde * pde_loss + \
                self.lambda_bc * (loss_Smax) + \
                self.lambda_tc * loss_T


        return total, {
            "pde": pde_loss.item(), "S0": loss_S0.item(),
            "Smax": loss_Smax.item(), "T": loss_T.item(), "total": total.item()
        }
