#!/usr/bin/env python3
"""
Simulation numérique simplifiée pour le résumé critique de
"Optimal Execution with Stochastic Delay" (Cartea & Sánchez-Betancourt, 2023).

On isole la source de valeur des MLOs : le gain par trade vient de la
distribution asymétrique des flickers. Un MLO capture les price improvements
(flicker > 0) et évite le slippage (flicker < 0 → missed).

Figures produites :
  1. Surperformance RLOS vs benchmarks en fonction de la latence
  2. Effet du paramètre d'urgence phi
  3. Impact de la distribution de latence (exponentielle vs Gamma)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

np.random.seed(42)

# ============================================================
# Paramètres (Table 4 de l'article, fenêtre 9h-9h10)
# ============================================================
S0 = 1.09520
sigma = 2.1e-4
T = 6.0
M = 10
tick = 1e-5

# Flickers (eq. 3.2) — Table 4 reporte 1/eta (mislabeled eta), cf. Table 11
p_plus = 0.05
p_minus = 0.08
p0 = 1.0 - p_plus - p_minus  # = 0.87
eta_plus = 1.0 / 1.87e-5   # E[improvement] = 1/eta_+ = 1.87e-5 ~ 1.87 ticks
eta_minus = 1.0 / 2.66e-5  # E[slippage]    = 1/eta_- = 2.66e-5 ~ 2.66 ticks

rho = 3e-6           # transaction cost rate (c = 3 $/€M)
a_penalty = 1.80e-5  # terminal penalty for walking the book


def sample_flickers(n):
    """Tirage vectorisé de n flickers selon la loi mixte (eq. 3.2)."""
    u = np.random.rand(n)
    flickers = np.zeros(n)
    mask_plus = (u >= p0) & (u < p0 + p_plus)
    mask_minus = u >= p0 + p_plus
    flickers[mask_plus] = np.random.exponential(1.0 / eta_plus, mask_plus.sum())
    flickers[mask_minus] = -np.random.exponential(1.0 / eta_minus, mask_minus.sum())
    return flickers


def simulate_rlos_value(mean_latency, phi, n_sims=50000, latency_shape=1):
    """
    Simule la valeur totale de la stratégie RLOS simplifiée.

    Paramètres
    ----------
    mean_latency : float
        Latence moyenne (en secondes).
    phi : float
        Paramètre d'urgence (running penalty φ q²).
    n_sims : int
        Nombre de trajectoires Monte Carlo.
    latency_shape : int
        Paramètre de forme k de la distribution Gamma(k, mean/k).
        k=1 → exponentielle (hypothèse de l'article).
        k>1 → distribution plus concentrée (critique Section 4.1).

    Retourne
    --------
    total_cash_gain : ndarray (n_sims,)
        Gain en cash relatif à S0 pour chaque trajectoire.
    stats : dict
        Statistiques moyennes (tentatives, fills, ratio spéculatif).
    """
    total_cash_gain = np.zeros(n_sims)
    total_attempts = 0
    total_fills = 0
    total_spec_attempts = 0
    total_spec_fills = 0

    # Pré-calcul du scale Gamma
    gamma_scale = mean_latency / latency_shape

    for sim in range(n_sims):
        cash = 0.0
        q = M
        t = 0.0

        while q > 0 and t < T:
            # Latence : Gamma(k, mean/k) — exponentielle si k=1
            delay = np.random.gamma(latency_shape, gamma_scale)
            notif_t = t + delay

            if notif_t >= T:
                break

            flicker = sample_flickers(1)[0]
            total_attempts += 1

            # Décision spéculatif vs normal (heuristique)
            time_left = T - notif_t
            expected_attempts_left = time_left / mean_latency
            slack = expected_attempts_left / max(q, 1)

            # Un trader patient (phi≈0) avec du slack → MLO spéculatif
            # Un trader impatient (phi grand) → MLO normal
            p_spec = max(0.0, min(1.0, (slack - 1.5) / slack)) * np.exp(-phi * 500)

            if np.random.rand() < p_spec:
                # MLO spéculatif (limit > S_t) : rempli ssi flicker > 0
                total_spec_attempts += 1
                if flicker > 0:
                    cash += flicker
                    q -= 1
                    total_fills += 1
                    total_spec_fills += 1
                # Sinon missed → on continue
            else:
                # MLO normal (limit ≈ S_t) : rempli sauf slippage extrême
                if flicker >= -3 * tick:
                    cash += flicker
                    q -= 1
                    total_fills += 1

            t = notif_t

        # Lots restants liquidés au terminal avec pénalité (walking the book)
        if q > 0 and q > 1:
            cash -= a_penalty * q * (q - 1)

        total_cash_gain[sim] = cash

    stats = {
        "mean_attempts": total_attempts / n_sims,
        "mean_fills": total_fills / n_sims,
        "spec_ratio": total_spec_attempts / max(total_attempts, 1),
        "spec_fill_rate": total_spec_fills / max(total_spec_attempts, 1),
    }
    return total_cash_gain, stats


def simulate_twap_value(n_sims=50000):
    """
    TWAP : envoie M MOs à intervalles réguliers.
    Les MOs n'ont pas de price protection → subissent le flicker complet.
    """
    # Vectorisé : chaque sim = somme de M flickers
    all_flickers = sample_flickers(n_sims * M).reshape(n_sims, M)
    return all_flickers.sum(axis=1)


def simulate_enow_value(n_sims=50000):
    """
    ENOW : 1 MO au temps 0, latence nulle → gain = 0 par rapport à S0.
    """
    return np.zeros(n_sims)


# ============================================================
# Simulation 1 : Performance vs latence moyenne (phi = 0)
# ============================================================
print("Simulation 1 : Performance vs latence moyenne (phi = 0)")
print("=" * 60)

latencies_ms = np.array([10, 20, 30, 50, 70, 90, 120])
n_sims = 50000
V0 = M * S0

perf_vs_twap = np.zeros(len(latencies_ms))
perf_vs_enow = np.zeros(len(latencies_ms))

# TWAP et ENOW ne dépendent pas de la latence
twap_gains = simulate_twap_value(n_sims=n_sims)
enow_gains = simulate_enow_value(n_sims=n_sims)
twap_mean = np.mean(twap_gains)
enow_mean = np.mean(enow_gains)

print(f"  TWAP : gain moyen par flickers = {twap_mean/V0*1e6:+.3f} $/€M")
print(f"  ENOW : gain moyen = {enow_mean/V0*1e6:+.3f} $/€M")
print()

for i, lat_ms in enumerate(latencies_ms):
    mean_lat = lat_ms * 1e-3
    rlos_gains, stats = simulate_rlos_value(mean_lat, phi=0.0, n_sims=n_sims)

    perf_vs_twap[i] = (np.mean(rlos_gains) - twap_mean) / V0 * 1e6
    perf_vs_enow[i] = (np.mean(rlos_gains) - enow_mean) / V0 * 1e6

    print(f"  Latence {lat_ms:3d} ms : vs TWAP = {perf_vs_twap[i]:+6.2f}, "
          f"vs ENOW = {perf_vs_enow[i]:+6.2f} $/€M  |  "
          f"tentatives = {stats['mean_attempts']:5.1f}, "
          f"spéc = {stats['spec_ratio']:.0%}, "
          f"fill spéc = {stats['spec_fill_rate']:.1%}")

# --- Figure 1 ---
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(latencies_ms, perf_vs_twap, "o-", color="royalblue",
         linewidth=2, markersize=7, label="RLOS vs TWAP", zorder=5)
ax1.plot(latencies_ms, perf_vs_enow, "s--", color="firebrick",
         linewidth=2, markersize=7, label="RLOS vs ENOW", zorder=5)
ax1.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

tc_value = rho * 1e6  # = 3 $/€M
ax1.axhspan(-tc_value, tc_value, alpha=0.08, color="green",
            label=f"Frais de transaction (±{tc_value:.0f} $/€M)")

ax1.set_xlabel("Latence moyenne (ms)")
ax1.set_ylabel(r"Surperformance ($ / €M échangé)")
ax1.set_title(r"Surperformance de RLOS ($\varphi = 0$, trader patient)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence.pdf")
fig1.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence.png")
print("\n  -> fig_latence.pdf sauvegardée\n")


# ============================================================
# Simulation 2 : Effet du paramètre d'urgence phi
# ============================================================
print("Simulation 2 : Effet de phi (latence = 30 ms)")
print("=" * 60)

phi_values = np.array([0, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
mean_lat_30 = 0.030

perf_phi = np.zeros(len(phi_values))

for i, phi in enumerate(phi_values):
    rlos_gains, stats = simulate_rlos_value(mean_lat_30, phi=phi, n_sims=n_sims)
    perf_phi[i] = (np.mean(rlos_gains) - twap_mean) / V0 * 1e6
    print(f"  phi = {phi:.1e} : RLOS - TWAP = {perf_phi[i]:+6.2f} $/€M  |  "
          f"spéc = {stats['spec_ratio']:.0%}")

# --- Figure 2 ---
phi_labels = ["0", r"$5{\cdot}10^{-6}$", r"$10^{-5}$", r"$5{\cdot}10^{-5}$",
              r"$10^{-4}$", r"$5{\cdot}10^{-4}$", r"$10^{-3}$", r"$5{\cdot}10^{-3}$"]

fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
x_pos = np.arange(len(phi_values))
colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in perf_phi]
ax2.bar(x_pos, perf_phi, color=colors, edgecolor="black",
        linewidth=0.5, alpha=0.85, width=0.55)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(phi_labels, fontsize=9)
ax2.set_xlabel(r"Paramètre d'urgence $\varphi$")
ax2.set_ylabel(r"RLOS $-$ TWAP ($ / €M)")
ax2.set_title(r"Effet de $\varphi$ sur la surperformance (latence = 30 ms)")
ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
ax2.grid(True, alpha=0.3, axis="y")

ymax = max(perf_phi)
ymin = min(perf_phi)
yrange = ymax - ymin if ymax != ymin else 1

if perf_phi[0] > 0:
    ax2.annotate("Patient\n(MLOs spéculatifs)", xy=(0, perf_phi[0]),
                 xytext=(1.5, perf_phi[0] + yrange * 0.15),
                 fontsize=9, ha="center", color="#264653",
                 arrowprops=dict(arrowstyle="->", color="#264653", lw=1.2))

idx_last = len(phi_values) - 1
ax2.annotate("Impatient\n(pas de MLOs spéculatifs)",
             xy=(idx_last, perf_phi[idx_last]),
             xytext=(idx_last - 1.5, perf_phi[idx_last] - yrange * 0.25),
             fontsize=9, ha="center", color="#9b2226",
             arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.2))

fig2.tight_layout()
fig2.savefig("/Users/danallouche/Documents/Cycle of conf/fig_phi.pdf")
fig2.savefig("/Users/danallouche/Documents/Cycle of conf/fig_phi.png")
print("\n  -> fig_phi.pdf sauvegardée\n")


# ============================================================
# Simulation 3 : Impact de la distribution de latence
# ============================================================
print("Simulation 3 : Exponentielle vs Gamma (phi = 0)")
print("=" * 60)
print("  Critique principale : l'article suppose une latence exponentielle")
print("  (sans mémoire). On teste avec Gamma(k, mu/k) — même moyenne,")
print("  variance divisée par k.\n")

shapes = {"Exp (k=1)": 1, "Gamma (k=5)": 5, "Gamma (k=20)": 20}
latencies_ms_3 = np.array([10, 20, 30, 50, 70, 90, 120])

perf_by_shape = {}

for label, k in shapes.items():
    perf = np.zeros(len(latencies_ms_3))
    for i, lat_ms in enumerate(latencies_ms_3):
        mean_lat = lat_ms * 1e-3
        rlos_gains, stats = simulate_rlos_value(
            mean_lat, phi=0.0, n_sims=n_sims, latency_shape=k
        )
        perf[i] = (np.mean(rlos_gains) - twap_mean) / V0 * 1e6

    perf_by_shape[label] = perf
    print(f"  {label:15s} : ", end="")
    print("  ".join(f"{lat_ms}ms={p:+.2f}" for lat_ms, p in
                    zip(latencies_ms_3, perf)))

# --- Figure 3 ---
fig3, ax3 = plt.subplots(figsize=(7, 4.5))

styles = {"Exp (k=1)": ("o-", "royalblue"),
          "Gamma (k=5)": ("s--", "#e9c46a"),
          "Gamma (k=20)": ("D:", "#e76f51")}

for label, perf in perf_by_shape.items():
    fmt, color = styles[label]
    cv = 1 / np.sqrt(shapes[label])
    ax3.plot(latencies_ms_3, perf, fmt, color=color, linewidth=2,
             markersize=7, label=f"{label}  (CV = {cv:.2f})", zorder=5)

ax3.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
ax3.axhspan(-tc_value, tc_value, alpha=0.08, color="green",
            label=f"Frais de transaction (±{tc_value:.0f} $/€M)")

ax3.set_xlabel("Latence moyenne (ms)")
ax3.set_ylabel(r"RLOS $-$ TWAP ($ / €M)")
ax3.set_title("Impact de la distribution de latence sur la surperformance")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_dist.pdf")
fig3.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_dist.png")
print("\n  -> fig_latence_dist.pdf sauvegardée\n")

print("=" * 60)
print("Toutes les simulations terminées avec succès.")
