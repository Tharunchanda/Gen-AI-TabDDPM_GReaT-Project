"""
evaluate_synthetic.py
─────────────────────
Computes and visualises evaluation metrics for TabDDPM synthetic data:
  1. Wasserstein distance        – per numerical feature
  2. Jensen-Shannon divergence   – per categorical feature
  3. L2 distance (correlation)   – between real & synthetic correlation matrices
  4. Density & Coverage          – using k-NN in normalised space

Outputs
  exp/churn/evaluation/metrics.json          – all scalar metrics
  exp/churn/evaluation/wasserstein.png
  exp/churn/evaluation/jsd.png
  exp/churn/evaluation/correlation_real.png
  exp/churn/evaluation/correlation_synthetic.png
  exp/churn/evaluation/correlation_diff.png
  exp/churn/evaluation/density_coverage.png
  exp/churn/evaluation/numerical_distributions.png
  exp/churn/evaluation/categorical_distributions.png
  exp/churn/evaluation/summary_dashboard.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS  — adjust if your layout differs
# ═══════════════════════════════════════════════════════════════════════════════
REAL_DATA_PATH = "data/churn"
SYNTHETIC_PATH = "exp/churn/synthetic"
OUT_DIR        = "exp/churn/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = [
    "Income", "Credit Score", "Credit History Length",
    "Outstanding Loans", "Balance", "NumOfProducts",
    "NumComplaints", "Number of Dependents", "Customer Tenure",
]
CAT_COLS = [
    "Gender", "Marital Status",
    "Education Level", "Customer Segment",
    "Preferred Communication Channel",
]
TARGET_COL = "Churn Flag"

# ── Aesthetic constants ───────────────────────────────────────────────────────
REAL_COLOR  = "#2563EB"   # blue
SYN_COLOR   = "#F59E0B"   # amber
GOOD_COLOR  = "#10B981"   # emerald
BAD_COLOR   = "#EF4444"   # red
BG_COLOR    = "#0F172A"   # dark navy
CARD_COLOR  = "#1E293B"
TEXT_COLOR  = "#F1F5F9"
MUTED_COLOR = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    CARD_COLOR,
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED_COLOR,
    "ytick.color":       MUTED_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        "#1E293B",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "axes.titlecolor":   TEXT_COLOR,
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data...")

X_num_real = np.load(os.path.join(REAL_DATA_PATH, "X_num_train.npy"),      allow_pickle=True).astype(np.float32)
X_cat_real = np.load(os.path.join(REAL_DATA_PATH, "X_cat_train.npy"),      allow_pickle=True)
y_real     = np.load(os.path.join(REAL_DATA_PATH, "y_train.npy"),          allow_pickle=True)

X_num_syn  = np.load(os.path.join(SYNTHETIC_PATH, "X_num_synthetic.npy"),  allow_pickle=True).astype(np.float32)
X_cat_syn  = np.load(os.path.join(SYNTHETIC_PATH, "X_cat_synthetic.npy"),  allow_pickle=True)
y_syn      = np.load(os.path.join(SYNTHETIC_PATH, "y_synthetic.npy"),      allow_pickle=True)

# Build DataFrames for convenience
df_real = pd.DataFrame(X_num_real, columns=NUM_COLS)
df_syn  = pd.DataFrame(X_num_syn,  columns=NUM_COLS)
for i, col in enumerate(CAT_COLS):
    df_real[col] = X_cat_real[:, i]
    df_syn[col]  = X_cat_syn[:, i]
df_real[TARGET_COL] = y_real
df_syn[TARGET_COL]  = y_syn

print(f"  Real:      {len(df_real):,} rows")
print(f"  Synthetic: {len(df_syn):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. WASSERSTEIN DISTANCE  (per numerical feature)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing Wasserstein distances...")

wasserstein_scores = {}
for col in NUM_COLS:
    r = df_real[col].dropna().values
    s = df_syn[col].dropna().values
    # Normalise to [0,1] so distances are comparable across features
    lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
    if hi > lo:
        r_n = (r - lo) / (hi - lo)
        s_n = (s - lo) / (hi - lo)
    else:
        r_n, s_n = r, s
    wasserstein_scores[col] = float(wasserstein_distance(r_n, s_n))

mean_wd = float(np.mean(list(wasserstein_scores.values())))
print(f"  Mean Wasserstein distance: {mean_wd:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax.set_facecolor(CARD_COLOR)

cols_sorted = sorted(wasserstein_scores, key=wasserstein_scores.get)
vals        = [wasserstein_scores[c] for c in cols_sorted]
colors      = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals]

bars = ax.barh(cols_sorted, vals, color=colors, height=0.6, edgecolor="#0F172A", linewidth=0.5)
for bar, val in zip(bars, vals):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8, color=TEXT_COLOR)

ax.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=1, alpha=0.6, label="Good (<0.05)")
ax.axvline(0.10, color=SYN_COLOR,  linestyle="--", linewidth=1, alpha=0.6, label="Acceptable (<0.10)")
ax.set_xlabel("Normalised Wasserstein Distance")
ax.set_title(f"Wasserstein Distance per Numerical Feature   |   Mean = {mean_wd:.4f}", pad=12)
ax.legend(fontsize=8, framealpha=0.2)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wasserstein.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved wasserstein.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. JENSEN-SHANNON DIVERGENCE  (per categorical feature)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing Jensen-Shannon divergences...")

def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD between two probability distributions (already normalised)."""
    p, q = np.array(p, dtype=np.float64), np.array(q, dtype=np.float64)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

jsd_scores = {}
for col in CAT_COLS:
    all_cats = sorted(set(df_real[col].astype(str).unique()) |
                      set(df_syn[col].astype(str).unique()))
    r_counts = df_real[col].astype(str).value_counts()
    s_counts = df_syn[col].astype(str).value_counts()
    p = np.array([r_counts.get(c, 0) for c in all_cats], dtype=float)
    q = np.array([s_counts.get(c, 0) for c in all_cats], dtype=float)
    jsd_scores[col] = jsd(p, q)

mean_jsd = float(np.mean(list(jsd_scores.values())))
print(f"  Mean JSD: {mean_jsd:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG_COLOR)
ax.set_facecolor(CARD_COLOR)

cols_sorted_j = sorted(jsd_scores, key=jsd_scores.get)
vals_j        = [jsd_scores[c] for c in cols_sorted_j]
colors_j      = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_j]

bars = ax.barh(cols_sorted_j, vals_j, color=colors_j, height=0.5, edgecolor="#0F172A", linewidth=0.5)
for bar, val in zip(bars, vals_j):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, color=TEXT_COLOR)

ax.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=1, alpha=0.6, label="Good (<0.05)")
ax.axvline(0.10, color=SYN_COLOR,  linestyle="--", linewidth=1, alpha=0.6, label="Acceptable (<0.10)")
ax.set_xlabel("Jensen-Shannon Divergence")
ax.set_title(f"JSD per Categorical Feature   |   Mean = {mean_jsd:.4f}", pad=12)
ax.legend(fontsize=8, framealpha=0.2)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "jsd.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved jsd.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION MATRIX L2 DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing correlation matrix L2 distance...")

corr_real = df_real[NUM_COLS].corr().values
corr_syn  = df_syn[NUM_COLS].corr().values
corr_diff = corr_real - corr_syn
l2_corr   = float(np.linalg.norm(corr_diff, ord="fro"))
print(f"  Frobenius L2 distance between correlation matrices: {l2_corr:.4f}")

# Custom diverging colormap
cmap_corr = LinearSegmentedColormap.from_list(
    "corr", ["#1D4ED8", "#0F172A", "#DC2626"], N=256
)
cmap_diff = LinearSegmentedColormap.from_list(
    "diff", ["#059669", "#0F172A", "#DC2626"], N=256
)

def plot_corr(matrix, title, filename, cmap, vmin=-1, vmax=1):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(NUM_COLS)))
    ax.set_yticks(range(len(NUM_COLS)))
    short = [c.replace(" ", "\n") for c in NUM_COLS]
    ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short, fontsize=7)
    for i in range(len(NUM_COLS)):
        for j in range(len(NUM_COLS)):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.4 else MUTED_COLOR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=MUTED_COLOR)
    ax.set_title(title, pad=14, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

plot_corr(corr_real, "Correlation Matrix — Real Data",      "correlation_real.png",      cmap_corr)
plot_corr(corr_syn,  "Correlation Matrix — Synthetic Data", "correlation_synthetic.png", cmap_corr)
plot_corr(corr_diff, f"Correlation Difference (Real − Syn)  |  L2 = {l2_corr:.4f}",
          "correlation_diff.png", cmap_diff, vmin=-1, vmax=1)
print("  Saved correlation_real.png, correlation_synthetic.png, correlation_diff.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DENSITY & COVERAGE  (k-NN based, Naeem et al. 2020)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing Density & Coverage (k-NN)...")

K_DENSITY = 5   # number of nearest neighbours

# Normalise numerical features to [0,1]
scaler = MinMaxScaler()
R = scaler.fit_transform(X_num_real)
S = scaler.transform(X_num_syn)

# Sub-sample for speed if very large
MAX_EVAL = 5000
if len(R) > MAX_EVAL:
    idx_r = np.random.choice(len(R), MAX_EVAL, replace=False)
    R_eval = R[idx_r]
else:
    R_eval = R
if len(S) > MAX_EVAL:
    idx_s = np.random.choice(len(S), MAX_EVAL, replace=False)
    S_eval = S[idx_s]
else:
    S_eval = S

print(f"  Using {len(R_eval):,} real and {len(S_eval):,} synthetic samples for k-NN eval...")

# Pairwise distances
RR = cdist(R_eval, R_eval, metric="euclidean")
np.fill_diagonal(RR, np.inf)
knn_radii_real = np.partition(RR, K_DENSITY, axis=1)[:, K_DENSITY - 1]  # k-th NN distance

SR = cdist(S_eval, R_eval, metric="euclidean")  # syn → real

# Coverage: fraction of real points that have at least one synthetic neighbour within radius
coverage_hits = np.any(SR.T < knn_radii_real[:, None], axis=1)
coverage = float(coverage_hits.mean())

# Density: average number of real k-NN spheres a synthetic point falls into
density_vals = []
for s_idx in range(len(S_eval)):
    count = np.sum(SR[s_idx] < knn_radii_real)
    density_vals.append(count / K_DENSITY)
density = float(np.mean(density_vals))

print(f"  Density:  {density:.4f}   (ideal ≈ 1.0)")
print(f"  Coverage: {coverage:.4f}  (ideal ≈ 1.0)")

# ── Plot density & coverage ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG_COLOR)

for ax, metric, val, label in zip(
    axes,
    ["Density", "Coverage"],
    [density, coverage],
    ["Ideal ≈ 1.0\n>1 → mode collapse\n<1 → under-coverage",
     "Ideal ≈ 1.0\n<1 → real regions not covered"]
):
    ax.set_facecolor(CARD_COLOR)
    color = GOOD_COLOR if 0.8 <= val <= 1.2 else (SYN_COLOR if 0.6 <= val <= 1.4 else BAD_COLOR)

    # Gauge-style bar
    ax.barh([0], [1.5], color="#334155", height=0.4, edgecolor="none")
    ax.barh([0], [min(val, 1.5)], color=color, height=0.4, edgecolor="none")
    ax.axvline(1.0, color=TEXT_COLOR, linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlim(0, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(metric, fontsize=14, pad=10)
    ax.text(val / 2 if val < 1.5 else 0.75, 0,
            f"{val:.4f}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")
    ax.text(0.98, -0.35, label, ha="right", va="top",
            fontsize=7.5, color=MUTED_COLOR, transform=ax.transData)
    ax.grid(axis="x", alpha=0.2)

fig.suptitle("Density & Coverage  (k-NN, k=5)", fontsize=13, y=1.02, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "density_coverage.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved density_coverage.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. NUMERICAL DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting numerical distributions...")

n_cols_plot = 3
n_rows_plot = int(np.ceil(len(NUM_COLS) / n_cols_plot))
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(16, n_rows_plot * 3.5), facecolor=BG_COLOR)
axes = axes.flatten()

for i, col in enumerate(NUM_COLS):
    ax = axes[i]
    ax.set_facecolor(CARD_COLOR)
    r_vals = df_real[col].dropna().values
    s_vals = df_syn[col].dropna().values
    lo = min(np.percentile(r_vals, 1), np.percentile(s_vals, 1))
    hi = max(np.percentile(r_vals, 99), np.percentile(s_vals, 99))
    bins = np.linspace(lo, hi, 40)
    ax.hist(r_vals, bins=bins, density=True, alpha=0.55,
            color=REAL_COLOR, label="Real", edgecolor="none")
    ax.hist(s_vals, bins=bins, density=True, alpha=0.55,
            color=SYN_COLOR,  label="Synthetic", edgecolor="none")
    wd = wasserstein_scores[col]
    color_wd = GOOD_COLOR if wd < 0.05 else (SYN_COLOR if wd < 0.1 else BAD_COLOR)
    ax.set_title(f"{col}\nWD = {wd:.4f}", fontsize=9, color=color_wd)
    ax.set_ylabel("Density", fontsize=7)
    ax.grid(alpha=0.15)
    if i == 0:
        ax.legend(fontsize=7, framealpha=0.2)

for j in range(len(NUM_COLS), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Numerical Feature Distributions  —  Real vs Synthetic",
             fontsize=14, y=1.01, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "numerical_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved numerical_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CATEGORICAL DISTRIBUTION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting categorical distributions...")

n_cat = len(CAT_COLS)
fig, axes = plt.subplots(1, n_cat, figsize=(5 * n_cat, 5), facecolor=BG_COLOR)
if n_cat == 1:
    axes = [axes]

for ax, col in zip(axes, CAT_COLS):
    ax.set_facecolor(CARD_COLOR)
    all_cats = sorted(set(df_real[col].astype(str).unique()) |
                      set(df_syn[col].astype(str).unique()))
    r_freq = df_real[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0)
    s_freq = df_syn[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0)

    x = np.arange(len(all_cats))
    w = 0.38
    ax.bar(x - w/2, r_freq.values, width=w, color=REAL_COLOR,
           alpha=0.85, label="Real",      edgecolor="#0F172A", linewidth=0.5)
    ax.bar(x + w/2, s_freq.values, width=w, color=SYN_COLOR,
           alpha=0.85, label="Synthetic", edgecolor="#0F172A", linewidth=0.5)

    short_cats = [c[:14] + ("…" if len(c) > 14 else "") for c in all_cats]
    ax.set_xticks(x)
    ax.set_xticklabels(short_cats, rotation=35, ha="right", fontsize=7)
    jv = jsd_scores[col]
    color_jv = GOOD_COLOR if jv < 0.05 else (SYN_COLOR if jv < 0.1 else BAD_COLOR)
    ax.set_title(f"{col}\nJSD = {jv:.4f}", fontsize=9, color=color_jv)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    if col == CAT_COLS[0]:
        ax.legend(fontsize=7, framealpha=0.2)

fig.suptitle("Categorical Feature Distributions  —  Real vs Synthetic",
             fontsize=13, y=1.02, color=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "categorical_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved categorical_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding summary dashboard...")

fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

# ── Top row: 4 KPI cards ──────────────────────────────────────────────────────
kpis = [
    ("Mean Wasserstein\nDistance",        mean_wd,  0.05,  0.10,  "lower"),
    ("Mean Jensen-Shannon\nDivergence",   mean_jsd, 0.05,  0.10,  "lower"),
    ("Correlation L2\nDistance",          l2_corr,  1.0,   2.0,   "lower"),
    ("Density",                           density,  0.8,   0.6,   "mid"),
]

for k, (title, val, good_thresh, ok_thresh, direction) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, k])
    ax.set_facecolor(CARD_COLOR)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    if direction == "lower":
        color = GOOD_COLOR if val <= good_thresh else (SYN_COLOR if val <= ok_thresh else BAD_COLOR)
        rating = "GOOD" if val <= good_thresh else ("OK" if val <= ok_thresh else "POOR")
    else:  # mid — closer to 1.0 is better
        diff = abs(val - 1.0)
        color = GOOD_COLOR if diff <= 0.2 else (SYN_COLOR if diff <= 0.4 else BAD_COLOR)
        rating = "GOOD" if diff <= 0.2 else ("OK" if diff <= 0.4 else "POOR")

    ax.text(0.5, 0.72, f"{val:.4f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.40, title, ha="center", va="center",
            fontsize=8, color=MUTED_COLOR, transform=ax.transAxes)
    ax.text(0.5, 0.15, rating, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "22",
                      edgecolor=color, linewidth=1))

# ── Middle row left: Wasserstein bar ─────────────────────────────────────────
ax_wd = fig.add_subplot(gs[1, :2])
ax_wd.set_facecolor(CARD_COLOR)
cols_s = sorted(wasserstein_scores, key=wasserstein_scores.get)
vals_s = [wasserstein_scores[c] for c in cols_s]
colors_s = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_s]
ax_wd.barh(cols_s, vals_s, color=colors_s, height=0.55, edgecolor="#0F172A", linewidth=0.3)
ax_wd.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
ax_wd.axvline(0.10, color=SYN_COLOR,  linestyle="--", linewidth=0.8, alpha=0.6)
ax_wd.set_title("Wasserstein Distance (Numerical)", fontsize=10)
ax_wd.grid(axis="x", alpha=0.2)

# ── Middle row right: JSD bar ─────────────────────────────────────────────────
ax_jsd = fig.add_subplot(gs[1, 2:])
ax_jsd.set_facecolor(CARD_COLOR)
cols_j2 = sorted(jsd_scores, key=jsd_scores.get)
vals_j2  = [jsd_scores[c] for c in cols_j2]
colors_j2 = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_j2]
ax_jsd.barh(cols_j2, vals_j2, color=colors_j2, height=0.45, edgecolor="#0F172A", linewidth=0.3)
ax_jsd.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
ax_jsd.axvline(0.10, color=SYN_COLOR,  linestyle="--", linewidth=0.8, alpha=0.6)
ax_jsd.set_title("Jensen-Shannon Divergence (Categorical)", fontsize=10)
ax_jsd.grid(axis="x", alpha=0.2)

# ── Bottom row: Density/Coverage gauges + legend ──────────────────────────────
for k, (metric, val) in enumerate([("Density", density), ("Coverage", coverage)]):
    ax = fig.add_subplot(gs[2, k])
    ax.set_facecolor(CARD_COLOR)
    color = GOOD_COLOR if 0.8 <= val <= 1.2 else (SYN_COLOR if 0.6 <= val <= 1.4 else BAD_COLOR)
    ax.barh([0], [1.5], color="#334155", height=0.35, edgecolor="none")
    ax.barh([0], [min(val, 1.5)], color=color, height=0.35, edgecolor="none")
    ax.axvline(1.0, color=TEXT_COLOR, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1.5); ax.set_yticks([])
    ax.text(min(val, 1.5) / 2, 0, f"{val:.4f}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.set_title(metric, fontsize=11)
    ax.grid(axis="x", alpha=0.2)

# ── Bottom row right: legend / colour guide ───────────────────────────────────
ax_leg = fig.add_subplot(gs[2, 2:])
ax_leg.set_facecolor(CARD_COLOR)
ax_leg.set_xticks([]); ax_leg.set_yticks([])
for spine in ax_leg.spines.values():
    spine.set_edgecolor("#334155")

legend_items = [
    (GOOD_COLOR, "GOOD",  "WD/JSD < 0.05  |  D&C ≈ 1.0"),
    (SYN_COLOR,  "OK",    "WD/JSD < 0.10  |  D&C ≈ 0.8"),
    (BAD_COLOR,  "POOR",  "WD/JSD ≥ 0.10  |  D&C < 0.6"),
    (REAL_COLOR, "Real",  "Real data distribution"),
    (SYN_COLOR,  "Synth", "Synthetic data distribution"),
]
for i, (c, label, desc) in enumerate(legend_items):
    y = 0.85 - i * 0.18
    ax_leg.add_patch(mpatches.FancyBboxPatch(
        (0.03, y - 0.06), 0.08, 0.10,
        boxstyle="round,pad=0.01", facecolor=c, edgecolor="none",
        transform=ax_leg.transAxes
    ))
    ax_leg.text(0.15, y, f"{label}:", fontsize=8, fontweight="bold",
                color=c, transform=ax_leg.transAxes, va="center")
    ax_leg.text(0.30, y, desc, fontsize=7.5, color=MUTED_COLOR,
                transform=ax_leg.transAxes, va="center")

ax_leg.set_title("Metric Colour Guide", fontsize=9)

fig.suptitle("Synthetic Data Quality Dashboard", fontsize=16,
             fontweight="bold", color=TEXT_COLOR, y=1.01)

plt.savefig(os.path.join(OUT_DIR, "summary_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print("  Saved summary_dashboard.png")

# 9. SAVE metrics.json
metrics = {
    "wasserstein": {
        "per_feature": wasserstein_scores,
        "mean":        mean_wd,
        "interpretation": "lower is better; <0.05 good, <0.10 acceptable"
    },
    "jensen_shannon_divergence": {
        "per_feature": jsd_scores,
        "mean":        mean_jsd,
        "interpretation": "lower is better; <0.05 good, <0.10 acceptable"
    },
    "correlation_l2": {
        "frobenius_distance": l2_corr,
        "interpretation": "lower is better; measures structure preservation"
    },
    "density": {
        "value": density,
        "k":     K_DENSITY,
        "interpretation": "ideal ≈ 1.0; >1 may indicate mode collapse, <1 under-coverage"
    },
    "coverage": {
        "value": coverage,
        "k":     K_DENSITY,
        "interpretation": "ideal ≈ 1.0; low means real regions not represented in synthetic"
    },
    "data_info": {
        "real_train_rows":  int(len(df_real)),
        "synthetic_rows":   int(len(df_syn)),
        "numerical_features": NUM_COLS,
        "categorical_features": CAT_COLS,
    }
}

json_path = os.path.join(OUT_DIR, "metrics.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  EVALUATION SUMMARY")
print("═" * 60)
print(f"  {'Metric':<35} {'Value':>10}  {'Rating'}")
print("─" * 60)

def rate(val, good, ok, direction="lower"):
    if direction == "lower":
        return "GOOD" if val <= good else ("OK" if val <= ok else "POOR")
    diff = abs(val - 1.0)
    return "GOOD" if diff <= good else ("OK" if diff <= ok else "POOR")

rows = [
    ("Mean Wasserstein Distance",       mean_wd,  0.05, 0.10, "lower"),
    ("Mean Jensen-Shannon Divergence",  mean_jsd, 0.05, 0.10, "lower"),
    ("Correlation L2 (Frobenius)",      l2_corr,  1.0,  2.0,  "lower"),
    ("Density  (k=5)",                  density,  0.2,  0.4,  "mid"),
    ("Coverage (k=5)",                  coverage, 0.2,  0.4,  "mid"),
]
for name, val, g, o, d in rows:
    r = rate(val, g, o, d)
    print(f"  {name:<35} {val:>10.4f}  {r}")

print("═" * 60)
print(f"\n  All outputs saved to: {OUT_DIR}/")
print(f"  Files: {sorted(os.listdir(OUT_DIR))}")