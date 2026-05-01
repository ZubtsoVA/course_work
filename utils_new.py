import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple


class CubicStretching:
    def __init__(self, B, S_min, S_max, alpha, chi=6.0, device='cpu'):
        self.B = B
        self.S_min = S_min
        self.S_max = S_max
        self.alpha = alpha
        self.chi = chi
        self.device = device

        # Решаем кубические уравнения для c1 и c2
        self.c1 = self._solve_cubic((B - S_min) / alpha)
        self.c2 = self._solve_cubic((B - S_max) / alpha)

    def _solve_cubic(self, d):
        # Решение: (1/chi)*c^3 + c + d = 0
        # Используем формулу Кардано или numpy
        import numpy as np
        coeffs = [1 / self.chi, 0, 1, d]
        roots = np.roots(coeffs)
        return float(roots[np.isreal(roots)].real[0])

    def transform(self, u):
        """u in [0,1] -> S in [S_min, S_max]"""
        u = torch.tensor(u, device=self.device)
        linear_comb = self.c2 * u + self.c1 * (1 - u)
        S = self.B + self.alpha * (
                (1 / self.chi) * linear_comb ** 3 + linear_comb
        )
        return S


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
            return torch.relu(S - self.K / self.S_max)
        else:
            return torch.relu(self.K / self.S_max - S)

    def far_field_bc(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Boundary condition at S = S_max."""
        if self.option == "call":
            t *= self.T
            return 1 - self.K * torch.exp(-self.r * t) / self.S_max

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


class BSLoss(nn.Module):
    def __init__(self, bs, S_grid_norm: torch.Tensor, t_grid_norm: torch.Tensor,
                 N_S: int, N_t: int, device: torch.device,
                 lambda_pde=1.0, lambda_bc=10.0, lambda_tc=10.0, lambda_violation=10.0):
        super().__init__()
        self.bs = bs
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_tc = lambda_tc
        self.lambda_violation = lambda_violation
        self.device = device
        self.alpha = 2 * bs.r / (bs.sigma ** 2)
        self.S_phys = (S_grid_norm * (bs.S_max - bs.S_min) + bs.S_min) / bs.S_max
        self.t_grid_norm = t_grid_norm.to(device)

        t_max_hat = (bs.sigma ** 2 / 2) * self.bs.T
        self.t_hat = (t_grid_norm * t_max_hat).to(device)
        self.t_max_hat = t_max_hat
        self.h_tau_hat = t_max_hat / (N_t - 1)
        # Шаги по безразмерной сетке
        self.h_t_hat = t_max_hat / (N_t - 1)
        self.mask_S0 = torch.zeros_like(S_grid_norm)
        self.mask_Smax = (self.S_phys > 0.95).float()
        self.mask_T = torch.zeros_like(S_grid_norm)
        self.mask_S0 = (self.S_phys < 0.04).float()

        # Внутренние точки (исключаем 2 граничных слоя по S и t для стабильных центральных разностей)
        self.mask_interior = (1.0 - self.mask_S0 - self.mask_Smax - self.mask_T).clamp(0, 1)
        self.mask_interior[:, :, :2, :] = 0.0
        self.mask_interior[:, :, -2:, :] = 0.0
        self.mask_interior[:, :, :, :2] = 0.0
        self.mask_interior[:, :, :, -2:] = 0.0

        # Количество активных точек для нормировки
        self.n_int = max(self.mask_interior.sum().item(), 1.0)
        self.n_S0 = max(self.mask_S0.sum().item(), 1.0)
        self.n_Smax = max(self.mask_Smax.sum().item(), 1.0)
        self.n_T = max(self.mask_T.sum().item(), 1.0)

    def masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tensor * mask).sum() / (mask.sum() + 1e-8)

    def forward(self, V: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        S_mid = self.S_phys[:, :, 1:-1, 1:-1]
        V_mid = V[:, :, 1:-1, 1:-1]
        V_left = V[:, :, :-2, 1:-1]
        V_right = V[:, :, 2:, 1:-1]
        S_left = self.S_phys[:, :, :-2, 1:-1]
        S_right = self.S_phys[:, :, 2:, 1:-1]

        h_L = S_mid - S_left
        h_R = S_right - S_mid
        dV_dS_f = (V_right - V_mid) / h_R.clamp(min=1e-8)
        dV_dS_b = (V_mid - V_left) / h_L.clamp(min=1e-8)
        dV_dS = 0.5 * (dV_dS_f + dV_dS_b)
        d2V_dS2 = 2.0 * (dV_dS_f - dV_dS_b) / (h_L + h_R).clamp(min=1e-8)
        dV_dtau = (V[:, :, :, 1:] - V[:, :, :, :-1]) / self.h_tau_hat
        dV_dtau_mid = dV_dtau[:, :, 1:-1, 1:]
        # 2. PDE Residual безразмерный
        rhs = S_mid ** 2 * d2V_dS2 + self.alpha * S_mid * dV_dS - self.alpha * V_mid
        pde_loss = ((dV_dtau_mid - rhs)**2).mean()
        #print(dV_dtau_mid.mean(), (S_mid ** 2 * d2V_dS2).mean(), (self.alpha * S_mid * dV_dS).mean(), self.alpha * V_mid.mean())
        #print(d2V_dS2.mean(), h_left.mean(), h_right.mean())
        # 3. Граничные условия

        t_phys = self.t_grid_norm * self.bs.T
        #print(self.bs.far_field_bc(self.S_grid_norm, t_phys ).max())
        loss_Smax = self.masked_mean((V - self.bs.far_field_bc(self.S_phys, t_phys)) ** 2, self.mask_Smax)
        #print(self.bs.far_field_bc(self.S_grid_norm, t_phys))
        loss_S0 = self.masked_mean(V.abs(), self.mask_S0)
        loss_boundary = loss_S0 + loss_Smax
        #violation_loss = ((torch.relu(dV_dtau - 1))**2).mean()
        #violation_loss = torch.mean(torch.relu(-dV_dS)) + torch.mean(torch.relu(-d2V_dS2))

        #violation_loss = (torch.mean(torch.relu(-dV_dS)) +
        #                  torch.mean(torch.relu(-d2V_dS2)) + torch.mean(torch.relu(dV_dtau_mid)))
        total = (self.lambda_pde * pde_loss +
                 self.lambda_bc * loss_boundary) #self.lambda_violation * violation_loss)

        #self.lambda_violation * (violation_loss))

        return total, {
            "pde": pde_loss.item(),
            "boundary": loss_boundary.item(),
            "T": 0,
            "total": total.item()
        }







