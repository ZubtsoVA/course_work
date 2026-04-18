"""
Physics-Informed CNN (PI-CNN) - Visualization and Main
======================================================
Модули 6, 7: Визуализация результатов и главный скрипт
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch_directml

from utils import BSParams
from model import PiCNN_BlackScholes, Trainer


# --------------------------
# 6. Visualisation & Evaluation
# --------------------------
def plot_results(trainer: Trainer, bs: BSParams):
    S_grid, t_grid, V_pred = trainer.predict_on_grid()
    S_np = S_grid
    t_np = t_grid
    print(V_pred)
    # Analytical solution
    V_true = bs.analytical_price(S_np, t_np)
    error = np.abs(V_pred - V_true)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"PI‑CNN for Black–Scholes ({bs.option.capitalize()})\n"
                 f"r={bs.r}, σ={bs.sigma}, K={bs.K}, T={bs.T}")

    # 1. Training loss
    ax = axes[0, 0]
    ax.semilogy(trainer.history["loss"], label="Total", linewidth=2)
    ax.semilogy(trainer.history["pde"], label="PDE", ls="--")
    ax.semilogy(trainer.history.get("Smax", []), label="BC Right", ls="-.")
    ax.semilogy(trainer.history.get("T", []), label="Terminal", ls=":")
    ax.set_xlabel("Epoch");
    ax.set_ylabel("Loss")
    ax.legend();
    ax.grid(True, alpha=0.3)
    ax.set_title("Training Loss")

    # 2. Predicted surface
    ax = axes[0, 1]
    cf = ax.contourf(S_np, t_np, V_pred, levels=40, cmap="viridis")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("S");
    ax.set_ylabel("t")
    ax.set_title("Predicted V(S,t)")

    # 3. Analytical surface
    ax = axes[0, 2]
    cf = ax.contourf(S_np, t_np, V_true, levels=40, cmap="viridis")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("S");
    ax.set_ylabel("t")
    ax.set_title("Analytical V(S,t)")

    # 4. Error map
    ax = axes[1, 0]
    cf = ax.contourf(S_np, t_np, error, levels=40, cmap="hot_r")
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel("S");
    ax.set_ylabel("t")
    ax.set_title(f"Absolute Error (max={error.max():.3f})")

    # 5. Terminal payoff (t = T)
    ax = axes[1, 1]
    idx_T = -1
    ax.plot(S_np[:, idx_T], V_pred[:, idx_T], 'b-', label="Predicted", linewidth=2)
    ax.plot(S_np[:, idx_T], V_true[:, idx_T], 'r--', label="Analytical")
    ax.set_xlabel("S");
    ax.set_ylabel("V")
    ax.set_title("Terminal Payoff (t = T)")
    ax.legend();
    ax.grid(True, alpha=0.3)

    # 6. Price at t = 0
    ax = axes[1, 2]
    idx_0 = 0
    ax.plot(S_np[:, idx_0], V_pred[:, idx_0], 'b-', label="Predicted", linewidth=2)
    ax.plot(S_np[:, idx_0], V_true[:, idx_0], 'r--', label="Analytical")
    ax.set_xlabel("S");
    ax.set_ylabel("V")
    ax.set_title("Option Price at t = 0")
    ax.legend();
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pi_cnn_bs_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig_val, ax_val = plt.subplots(1, 2, figsize=(12, 5))
    fig_val.suptitle("Interior Points Validation", fontsize=14, fontweight='bold')


    # Исключаем границы: S=0, S=S_max, t=0, t=T
    mask_interior = (S_np > bs.S_min + 1e-3) & (S_np < bs.S_max - 1e-3) & \
                    (t_np > 1e-3) & (t_np < bs.T - 1e-3)

    V_true_int = V_true[mask_interior]
    V_pred_int = V_pred[mask_interior]

    # 1. Scatter: Predicted vs Analytical
    ax_val[0].scatter(V_true_int.flatten(), V_pred_int.flatten(),
                      s=1, alpha=0.3, c='steelblue', label='Interior points')
    min_v, max_v = V_true.min(), V_true.max()
    ax_val[0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect match')
    ax_val[0].set_xlabel("Analytical Price")
    ax_val[0].set_ylabel("PiCNN Prediction")
    ax_val[0].set_title("Predicted vs Analytical")
    ax_val[0].legend(fontsize=9)
    ax_val[0].grid(True, alpha=0.3)
    ax_val[0].set_aspect('equal', 'box')

    # 2. Гистограмма относительной ошибки
    rel_error = np.abs(V_pred - V_true) / (np.abs(V_true) + 1e-8)
    ax_val[1].hist(rel_error[mask_interior].flatten() * 100,
                   bins=50, log=True, color='coral', edgecolor='black', alpha=0.7)
    ax_val[1].set_xlabel("Relative Error (%)")
    ax_val[1].set_ylabel("Frequency (log scale)")
    ax_val[1].set_title("Relative Error Distribution (Interior)")
    ax_val[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("pi_cnn_interior_validation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Статистика в консоль
    int_rel_err = rel_error[mask_interior].mean()
    int_abs_err = np.abs(V_pred - V_true)[mask_interior].mean()
    print(f"\nInterior Validation (excl. boundaries):")
    print(f"  Mean Relative Error : {int_rel_err * 100:.3f}%")
    print(f"  Mean Absolute Error : {int_abs_err:.4f}")
    print(f"  Points evaluated    : {mask_interior.sum()}")

    # Final metrics
    l2_error = np.sqrt(np.mean((V_pred - V_true) ** 2))
    linf_error = error.max()
    right_boundary_error = error[-1, 0]

    print("\nFinal metrics:")
    print(f"  L2 error       : {l2_error:.4e}")
    print(f"  L∞ error       : {linf_error:.4e}")
    print(f"  Relative L2    : {l2_error / np.sqrt(np.mean(V_true ** 2)):.4e}")
    print(f"  Right BC error : {right_boundary_error:.4f} (at S={bs.S_max}, t=0)")
    print(f"  V_true(S_max,0): {V_true[-1, 0]:.2f}")
    print(f"  V_pred(S_max,0): {V_pred[-1, 0]:.2f}")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --------------------------
# 7. Main
# --------------------------
if __name__ == "__main__":
    # Device selection
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"Using DirectML device: {torch_directml.device_name(0)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Black‑Scholes parameters
    bs = BSParams(
        r=0.05, sigma=0.20, K=100.0, T=1.0, S_max=700.0, option="call"
    )

    # Grid resolution (N_S × N_t)
    grid_shape = (256,128)


    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Model
    model = PiCNN_BlackScholes(activation="gelu", features=[32, 64, 128, 256]).to(device)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    model.apply(init_weights)
    # FIXED: Adjusted hyperparameters
    trainer = Trainer(
        model, bs, grid_shape=grid_shape,
        lr=1e-3, epochs=2100, device=device,  # More epochs
        lambda_pde=1.0, lambda_bc=10.0, lambda_tc=15.0, lambda_violation= 0.0
    )

    print("Training PI‑CNN for Black–Scholes")

    print()

    trainer.train(verbose_freq=300)

    print("\nGenerating plots...")
    plot_results(trainer, bs)
