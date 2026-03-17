#!/usr/bin/env python3
"""
Simplified numerical simulation for the critical review of
"Optimal Execution with Stochastic Delay" (Cartea & Sanchez-Betancourt, 2023).

We isolate the source of value of MLOs: the gain per trade comes from the
asymmetric distribution of flickers. An MLO captures price improvements
(flicker > 0) and avoids slippage (flicker < 0 -> missed).

Figures produced:
  1. RLOS outperformance vs benchmarks as a function of latency
  2. Effect of the urgency parameter phi
  3. Impact of the latency distribution (exponential vs Gamma)
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
# Parameters (Table 4 of the paper, 9:00am-9:10am window)
# ============================================================
S0 = 1.09520
sigma = 2.1e-4
T = 6.0
M = 10
tick = 1e-5

# Flickers (eq. 3.2) -- Table 4 reports 1/eta (mislabeled as eta), cf. Table 11
p_plus = 0.05
p_minus = 0.08
p0 = 1.0 - p_plus - p_minus  # = 0.87
eta_plus = 1.0 / 1.87e-5   # E[improvement] = 1/eta_+ = 1.87e-5 ~ 1.87 ticks
eta_minus = 1.0 / 2.66e-5  # E[slippage]    = 1/eta_- = 2.66e-5 ~ 2.66 ticks

rho = 3e-6           # transaction cost rate (c = 3 $/EUR M)
a_penalty = 1.80e-5  # terminal penalty for walking the book


def sample_flickers(n):
    """Vectorised sampling of n flickers from the mixture law (eq. 3.2)."""
    u = np.random.rand(n)
    flickers = np.zeros(n)
    mask_plus = (u >= p0) & (u < p0 + p_plus)
    mask_minus = u >= p0 + p_plus
    flickers[mask_plus] = np.random.exponential(1.0 / eta_plus, mask_plus.sum())
    flickers[mask_minus] = -np.random.exponential(1.0 / eta_minus, mask_minus.sum())
    return flickers


def simulate_rlos_value(mean_latency, phi, n_sims=50000, latency_shape=1):
    """
    Simulate the total value of the simplified RLOS strategy.

    Parameters
    ----------
    mean_latency : float
        Mean latency (in seconds).
    phi : float
        Urgency parameter (running penalty phi * q^2).
    n_sims : int
        Number of Monte Carlo trajectories.
    latency_shape : int
        Shape parameter k of the Gamma(k, mean/k) distribution.
        k=1 -> exponential (paper's assumption).
        k>1 -> more concentrated distribution (Section 4.1 critique).

    Returns
    -------
    total_cash_gain : ndarray (n_sims,)
        Cash gain relative to S0 for each trajectory.
    stats : dict
        Average statistics (attempts, fills, speculative ratio).
    """
    total_cash_gain = np.zeros(n_sims)
    total_attempts = 0
    total_fills = 0
    total_spec_attempts = 0
    total_spec_fills = 0

    # Pre-compute Gamma scale
    gamma_scale = mean_latency / latency_shape

    for sim in range(n_sims):
        cash = 0.0
        q = M
        t = 0.0

        while q > 0 and t < T:
            # Latency: Gamma(k, mean/k) -- exponential if k=1
            delay = np.random.gamma(latency_shape, gamma_scale)
            notif_t = t + delay

            if notif_t >= T:
                break

            flicker = sample_flickers(1)[0]
            total_attempts += 1

            # Speculative vs normal decision (heuristic)
            time_left = T - notif_t
            expected_attempts_left = time_left / mean_latency
            slack = expected_attempts_left / max(q, 1)

            # A patient trader (phi~0) with slack -> speculative MLO
            # An impatient trader (phi large) -> normal MLO
            p_spec = max(0.0, min(1.0, (slack - 1.5) / slack)) * np.exp(-phi * 500)

            if np.random.rand() < p_spec:
                # Speculative MLO (limit > S_t): filled iff flicker > 0
                total_spec_attempts += 1
                if flicker > 0:
                    cash += flicker
                    q -= 1
                    total_fills += 1
                    total_spec_fills += 1
                # Otherwise missed -> continue
            else:
                # Normal MLO (limit ~ S_t): filled unless extreme slippage
                if flicker >= -3 * tick:
                    cash += flicker
                    q -= 1
                    total_fills += 1

            t = notif_t

        # Remaining lots liquidated at terminal with penalty (walking the book)
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
    TWAP: sends M MOs at regular intervals.
    MOs have no price protection -> incur the full flicker.
    """
    # Vectorised: each sim = sum of M flickers
    all_flickers = sample_flickers(n_sims * M).reshape(n_sims, M)
    return all_flickers.sum(axis=1)


def simulate_enow_value(n_sims=50000):
    """
    ENOW: 1 MO at time 0, zero latency -> gain = 0 relative to S0.
    """
    return np.zeros(n_sims)


# ============================================================
# Simulation 1: Performance vs mean latency (phi = 0)
# ============================================================
print("Simulation 1: Performance vs mean latency (phi = 0)")
print("=" * 60)

latencies_ms = np.array([10, 20, 30, 50, 70, 90, 120])
n_sims = 50000
V0 = M * S0

perf_vs_twap = np.zeros(len(latencies_ms))
perf_vs_enow = np.zeros(len(latencies_ms))

# TWAP and ENOW do not depend on latency
twap_gains = simulate_twap_value(n_sims=n_sims)
enow_gains = simulate_enow_value(n_sims=n_sims)
twap_mean = np.mean(twap_gains)
enow_mean = np.mean(enow_gains)

print(f"  TWAP: mean flicker gain = {twap_mean/V0*1e6:+.3f} $/EUR M")
print(f"  ENOW: mean gain = {enow_mean/V0*1e6:+.3f} $/EUR M")
print()

for i, lat_ms in enumerate(latencies_ms):
    mean_lat = lat_ms * 1e-3
    rlos_gains, stats = simulate_rlos_value(mean_lat, phi=0.0, n_sims=n_sims)

    perf_vs_twap[i] = (np.mean(rlos_gains) - twap_mean) / V0 * 1e6
    perf_vs_enow[i] = (np.mean(rlos_gains) - enow_mean) / V0 * 1e6

    print(f"  Latency {lat_ms:3d} ms: vs TWAP = {perf_vs_twap[i]:+6.2f}, "
          f"vs ENOW = {perf_vs_enow[i]:+6.2f} $/EUR M  |  "
          f"attempts = {stats['mean_attempts']:5.1f}, "
          f"spec = {stats['spec_ratio']:.0%}, "
          f"spec fill = {stats['spec_fill_rate']:.1%}")

# --- Figure 1 ---
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(latencies_ms, perf_vs_twap, "o-", color="royalblue",
         linewidth=2, markersize=7, label="RLOS vs TWAP", zorder=5)
ax1.plot(latencies_ms, perf_vs_enow, "s--", color="firebrick",
         linewidth=2, markersize=7, label="RLOS vs ENOW", zorder=5)
ax1.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

tc_value = rho * 1e6  # = 3 $/EUR M
ax1.axhspan(-tc_value, tc_value, alpha=0.08, color="green",
            label=f"Transaction costs ($\\pm${tc_value:.0f} $/EUR M)")

ax1.set_xlabel("Mean latency (ms)")
ax1.set_ylabel(r"Outperformance ($ / EUR M traded)")
ax1.set_title(r"RLOS outperformance ($\varphi = 0$, patient trader)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_en.pdf")
fig1.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_en.png")
print("\n  -> fig_latence_en.pdf saved\n")


# ============================================================
# Simulation 2: Effect of the urgency parameter phi
# ============================================================
print("Simulation 2: Effect of phi (latency = 30 ms)")
print("=" * 60)

phi_values = np.array([0, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
mean_lat_30 = 0.030

perf_phi = np.zeros(len(phi_values))

for i, phi in enumerate(phi_values):
    rlos_gains, stats = simulate_rlos_value(mean_lat_30, phi=phi, n_sims=n_sims)
    perf_phi[i] = (np.mean(rlos_gains) - twap_mean) / V0 * 1e6
    print(f"  phi = {phi:.1e}: RLOS - TWAP = {perf_phi[i]:+6.2f} $/EUR M  |  "
          f"spec = {stats['spec_ratio']:.0%}")

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
ax2.set_xlabel(r"Urgency parameter $\varphi$")
ax2.set_ylabel(r"RLOS $-$ TWAP ($ / EUR M)")
ax2.set_title(r"Effect of $\varphi$ on outperformance (latency = 30 ms)")
ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
ax2.grid(True, alpha=0.3, axis="y")

ymax = max(perf_phi)
ymin = min(perf_phi)
yrange = ymax - ymin if ymax != ymin else 1

if perf_phi[0] > 0:
    ax2.annotate("Patient\n(speculative MLOs)", xy=(0, perf_phi[0]),
                 xytext=(1.5, perf_phi[0] + yrange * 0.15),
                 fontsize=9, ha="center", color="#264653",
                 arrowprops=dict(arrowstyle="->", color="#264653", lw=1.2))

idx_last = len(phi_values) - 1
ax2.annotate("Impatient\n(no speculative MLOs)",
             xy=(idx_last, perf_phi[idx_last]),
             xytext=(idx_last - 1.5, perf_phi[idx_last] - yrange * 0.25),
             fontsize=9, ha="center", color="#9b2226",
             arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.2))

fig2.tight_layout()
fig2.savefig("/Users/danallouche/Documents/Cycle of conf/fig_phi_en.pdf")
fig2.savefig("/Users/danallouche/Documents/Cycle of conf/fig_phi_en.png")
print("\n  -> fig_phi_en.pdf saved\n")


# ============================================================
# Simulation 3: Impact of the latency distribution
# ============================================================
print("Simulation 3: Exponential vs Gamma (phi = 0)")
print("=" * 60)
print("  Main critique: the paper assumes exponential latency")
print("  (memoryless). We test with Gamma(k, mu/k) -- same mean,")
print("  variance divided by k.\n")

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
    print(f"  {label:15s}: ", end="")
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
            label=f"Transaction costs ($\\pm${tc_value:.0f} $/EUR M)")

ax3.set_xlabel("Mean latency (ms)")
ax3.set_ylabel(r"RLOS $-$ TWAP ($ / EUR M)")
ax3.set_title("Impact of latency distribution on outperformance")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_dist_en.pdf")
fig3.savefig("/Users/danallouche/Documents/Cycle of conf/fig_latence_dist_en.png")
print("\n  -> fig_latence_dist_en.pdf saved\n")

print("=" * 60)
print("All simulations completed successfully.")
