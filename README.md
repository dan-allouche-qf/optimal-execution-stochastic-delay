# Optimal Execution with Stochastic Delay

Critical summary and Monte Carlo simulation of **"Optimal Execution with Stochastic Delay"** by Álvaro Cartea & Leandro Sánchez-Betancourt (*Finance and Stochastics*, 27:1–47, 2023).

Work by **Dan Allouche**, **Liam Abensour**, and **Corentin Srun** — M2 MASEF, Université Paris Dauphine.
Course: *Strategies and Actors of Portfolio Management* (Philippe Bergault).

## About the paper

A trader must liquidate an inventory of lots over a short horizon by sending **Marketable Limit Orders (MLOs)** in the presence of **stochastic latency**. The problem is formulated as a stochastic impulse control problem solved via a coupled HJBQVI system. The key insight is that patient traders use speculative MLOs — free short-lived options that capture price improvements while avoiding slippage — yielding an outperformance of 3–5 $/EUR M over standard benchmarks (TWAP, ENOW).

## Our contributions

- **Critical review** of the model's assumptions: exponential latency (memoryless), i.i.d. flickers, single pending order, zero price impact in backtests.
- **Monte Carlo simulation** (50,000 trajectories) reproducing the main results with a simplified heuristic for the speculative/normal MLO decision.
- **Numerical experiments**: effect of latency, urgency parameter φ, and latency distribution shape (Exponential vs Gamma).
- **Erratum on Table 4**: the column labelled η̂₊, η̂₋ actually reports 1/η̂ (mean flicker sizes, not rate parameters), as confirmed by cross-referencing with Table 11 of the paper.

## Repository contents

| File | Description |
|------|-------------|
| `Optimal_Execution_with_Stochastic_Delay.pdf` | Original research article |
| `Summary_FR.pdf` | Résumé critique (français, 7 pages) |
| `Summary_EN.pdf` | Critical summary (English, 7 pages) |
| `simulation_fr.py` | Python simulation — French comments & plots |
| `simulation_en.py` | Python simulation — English comments & plots |

## Generated figures

| Figure | Description |
|--------|-------------|
| `fig_latence.pdf` | RLOS outperformance vs mean latency |
| `fig_phi.pdf` | Effect of urgency parameter φ |
| `fig_latence_dist.pdf` | Exponential vs Gamma latency distributions |

## Usage

```bash
python simulation_en.py
```

Generates 3 figures (PDF + PNG) and prints simulation statistics to stdout.

**Dependencies:** `numpy`, `matplotlib`
