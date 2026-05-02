import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


REAL_DATA_PATH = "data/churn"
SYNTHETIC_PATH = "exp/churn/synthetic_cond1"
OUT_DIR = "exp/churn/evaluation_cond1"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = [
    "Income",
    "Credit Score",
    "Credit History Length",
    "Outstanding Loans",
    "Balance",
    "NumOfProducts",
    "NumComplaints",
    "Number of Dependents",
    "Customer Tenure",
]
TARGET_COL = "Churn Flag"
CAT_COLS = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Customer Segment",
    "Preferred Communication Channel",
]

K_DENSITY = 5
MAX_EVAL = 5000
LOWER_Q = 0.001
UPPER_Q = 0.999
EVAL_SEED = 0

# Color scheme for visualizations
# IEEE-friendly color scheme (print + digital safe)

BG_COLOR    = "#FFFFFF"   # white background
CARD_COLOR  = "#FFFFFF"   # no dark cards in papers
TEXT_COLOR  = "#000000"   # black text
MUTED_COLOR = "#6B7280"   # neutral gray

GOOD_COLOR  = "#009E73"   # green (improvement)
SYN_COLOR   = "#0072B2"   # blue (synthetic data)
BAD_COLOR   = "#D55E00"   # reddish-orange (degradation)
REAL_COLOR  = "#000000"   # black (real data)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def print_summary(metrics: dict) -> None:
    print("\nEVALUATION SUMMARY (COND1)")
    print("=" * 62)
    print(f"{'Metric':38s}{'Value':>12s}")
    print("-" * 62)
    print(f"{'Mean Wasserstein Distance':38s}{metrics['mean_wasserstein_distance']:12.4f}")
    print(f"{'Mean Jensen-Shannon Divergence':38s}{metrics['mean_jsd']:12.4f}")
    print(f"{'Correlation L2 (Frobenius)':38s}{metrics['correlation_l2_frobenius']:12.4f}")
    print(f"{'Density (k=5)':38s}{metrics['density_k5']:12.4f}")
    print(f"{'Coverage (k=5)':38s}{metrics['coverage_k5']:12.4f}")
    print(f"{'Constraint Violation Rate':38s}{metrics['constraint_violation_rate']:12.4f}")
    print(f"{'Constraint Violation Magnitude':38s}{metrics['constraint_violation_magnitude']:12.6f}")
    print("=" * 62)


def generate_visualizations(X_num_real, X_cat_real, X_num_syn, X_cat_syn, 
                           wasserstein_scores, jsd_scores, l2_corr, density, coverage,
                           NUM_COLS, CAT_COLS, OUT_DIR):
    """Generate PNG visualization files for Cond1 evaluation."""
    
    print("\nGenerating visualization PNG files...")
    
    # Convert to DataFrames for easier plotting
    df_real = pd.DataFrame(X_num_real, columns=NUM_COLS)
    df_syn = pd.DataFrame(X_num_syn, columns=NUM_COLS)
    
    # 1. Wasserstein bar chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    cols_s = sorted(wasserstein_scores, key=wasserstein_scores.get)
    vals_s = [wasserstein_scores[c] for c in cols_s]
    colors_s = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_s]
    ax.barh(cols_s, vals_s, color=colors_s, edgecolor="#0F172A", linewidth=0.5)
    ax.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(0.10, color=SYN_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Wasserstein Distance", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Per-Feature Wasserstein Distance", fontsize=12, color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "wasserstein.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved wasserstein.png")
    
    # 2. JSD bar chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)
    cols_j = sorted(jsd_scores, key=jsd_scores.get)
    vals_j = [jsd_scores[c] for c in cols_j]
    colors_j = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_j]
    ax.barh(cols_j, vals_j, color=colors_j, edgecolor="#0F172A", linewidth=0.5)
    ax.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(0.10, color=SYN_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Jensen-Shannon Divergence", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Per-Feature Jensen-Shannon Divergence", fontsize=12, color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "jsd.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved jsd.png")
    
    # 3. Density & Coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)
    for ax, metric, val in zip(axes, ["Density", "Coverage"], [density, coverage]):
        ax.set_facecolor(CARD_COLOR)
        color = GOOD_COLOR if 0.8 <= val <= 1.2 else (SYN_COLOR if 0.6 <= val <= 1.4 else BAD_COLOR)
        ax.barh([0], [1.5], color="#334155", height=0.4, edgecolor="none")
        ax.barh([0], [min(val, 1.5)], color=color, height=0.4, edgecolor="none")
        ax.axvline(1.0, color=TEXT_COLOR, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1.5)
        ax.set_yticks([])
        ax.set_xlabel("Score", fontsize=10, color=TEXT_COLOR)
        ax.set_title(metric, fontsize=12, color=TEXT_COLOR)
        ax.text(val / 2 if val < 1.5 else 0.75, 0, f"{val:.4f}", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Density & Coverage (k-NN, k=5)", fontsize=13, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "density_coverage.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved density_coverage.png")
    
    # 4. Numerical distributions
    n_cols_plot = 3
    n_rows_plot = int(np.ceil(len(NUM_COLS) / n_cols_plot))
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(16, n_rows_plot * 3.5), facecolor=BG_COLOR)
    axes = axes.flatten()
    for i, col in enumerate(NUM_COLS):
        ax = axes[i]
        ax.set_facecolor(CARD_COLOR)
        r_vals = df_real[col].dropna().values
        s_vals = df_syn[col].dropna().values
        lo = min(np.percentile(r_vals, 1), np.percentile(s_vals, 1))
        hi = max(np.percentile(r_vals, 99), np.percentile(s_vals, 99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(r_vals, bins=bins, density=True, alpha=0.55, color=REAL_COLOR, label="Real", edgecolor="none")
        ax.hist(s_vals, bins=bins, density=True, alpha=0.55, color=SYN_COLOR, label="Synthetic", edgecolor="none")
        wd = wasserstein_scores[col]
        color_wd = GOOD_COLOR if wd < 0.05 else (SYN_COLOR if wd < 0.1 else BAD_COLOR)
        ax.set_title(f"{col}\nWD = {wd:.4f}", fontsize=9, color=color_wd)
        ax.set_ylabel("Density", fontsize=7, color=TEXT_COLOR)
        ax.grid(alpha=0.15)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.2)
    for j in range(len(NUM_COLS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Feature Distributions — Real vs Synthetic", fontsize=14, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "numerical_distributions.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved numerical_distributions.png")
    
    # 5. Categorical distributions
    n_cat = len(CAT_COLS)
    fig, axes = plt.subplots(1, n_cat, figsize=(5 * n_cat, 5), facecolor=BG_COLOR)
    if n_cat == 1:
        axes = [axes]
    
    df_cat_real = pd.DataFrame(X_cat_real, columns=CAT_COLS)
    df_cat_syn = pd.DataFrame(X_cat_syn, columns=CAT_COLS)
    
    for ax, col in zip(axes, CAT_COLS):
        ax.set_facecolor(CARD_COLOR)
        all_cats = sorted(set(df_cat_real[col].astype(str).unique()) | set(df_cat_syn[col].astype(str).unique()))
        r_freq = df_cat_real[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0)
        s_freq = df_cat_syn[col].astype(str).value_counts(normalize=True).reindex(all_cats, fill_value=0)
        
        x = np.arange(len(all_cats))
        w = 0.38
        ax.bar(x - w/2, r_freq.values, width=w, color=REAL_COLOR, alpha=0.85, label="Real", edgecolor="#0F172A", linewidth=0.5)
        ax.bar(x + w/2, s_freq.values, width=w, color=SYN_COLOR, alpha=0.85, label="Synthetic", edgecolor="#0F172A", linewidth=0.5)
        
        short_cats = [c[:14] + ("…" if len(c) > 14 else "") for c in all_cats]
        ax.set_xticks(x)
        ax.set_xticklabels(short_cats, rotation=35, ha="right", fontsize=7)
        jv = jsd_scores[col]
        color_jv = GOOD_COLOR if jv < 0.05 else (SYN_COLOR if jv < 0.1 else BAD_COLOR)
        ax.set_title(f"{col}\nJSD = {jv:.4f}", fontsize=9, color=color_jv)
        ax.set_ylabel("Proportion", fontsize=7, color=TEXT_COLOR)
        ax.grid(axis="y", alpha=0.2)
        if col == CAT_COLS[0]:
            ax.legend(fontsize=7, framealpha=0.2)
    
    fig.suptitle("Categorical Feature Distributions — Real vs Synthetic", fontsize=13, color=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "categorical_distributions.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved categorical_distributions.png")
    
    # 6. Correlation heatmaps
    mean_wd = np.mean(list(wasserstein_scores.values()))
    mean_jsd = np.mean(list(jsd_scores.values()))
    
    corr_real = np.corrcoef(X_num_real, rowvar=False)
    corr_syn = np.corrcoef(X_num_syn, rowvar=False)
    corr_diff = corr_real - corr_syn
    
    for data, name in [(corr_real, "correlation_real"), 
                       (corr_syn, "correlation_synthetic"),
                       (corr_diff, "correlation_diff")]:
        fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG_COLOR)
        im = ax.imshow(data, cmap="RdBu_r" if name == "correlation_diff" else "coolwarm", aspect="auto", vmin=-1 if name != "correlation_diff" else None, vmax=1)
        ax.set_xticks(np.arange(len(NUM_COLS)))
        ax.set_yticks(np.arange(len(NUM_COLS)))
        ax.set_xticklabels(NUM_COLS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(NUM_COLS, fontsize=8)
        for i in range(len(NUM_COLS)):
            for j in range(len(NUM_COLS)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(data[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax)
        title_map = {"correlation_real": "Real Data Correlation", 
                    "correlation_synthetic": "Synthetic Data Correlation",
                    "correlation_diff": "Correlation Difference (Real - Synthetic)"}
        ax.set_title(title_map[name], fontsize=12, color=TEXT_COLOR, pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        print(f"  Saved {name}.png")
    
    # 7. Summary dashboard
    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)
    
    # KPI cards
    kpis = [
        ("Mean Wasserstein\nDistance", mean_wd, 0.05, 0.10, "lower"),
        ("Mean Jensen-Shannon\nDivergence", mean_jsd, 0.05, 0.10, "lower"),
        ("Correlation L2\nDistance", l2_corr, 1.0, 2.0, "lower"),
        ("Density", density, 0.8, 0.6, "mid"),
    ]
    
    for k, (title, val, good_thresh, ok_thresh, direction) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor(CARD_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        
        if direction == "lower":
            color = GOOD_COLOR if val <= good_thresh else (SYN_COLOR if val <= ok_thresh else BAD_COLOR)
            rating = "GOOD" if val <= good_thresh else ("OK" if val <= ok_thresh else "POOR")
        else:
            diff = abs(val - 1.0)
            color = GOOD_COLOR if diff <= 0.2 else (SYN_COLOR if diff <= 0.4 else BAD_COLOR)
            rating = "GOOD" if diff <= 0.2 else ("OK" if diff <= 0.4 else "POOR")
        
        ax.text(0.5, 0.72, f"{val:.4f}", ha="center", va="center", fontsize=20, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.40, title, ha="center", va="center", fontsize=8, color=MUTED_COLOR, transform=ax.transAxes)
        ax.text(0.5, 0.15, rating, ha="center", va="center", fontsize=9, fontweight="bold", color=color, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color + "22", edgecolor=color, linewidth=1))
    
    # Wasserstein bar
    ax_wd = fig.add_subplot(gs[1, :2])
    ax_wd.set_facecolor(CARD_COLOR)
    cols_s = sorted(wasserstein_scores, key=wasserstein_scores.get)
    vals_s = [wasserstein_scores[c] for c in cols_s]
    colors_s = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_s]
    ax_wd.barh(cols_s, vals_s, color=colors_s, height=0.55, edgecolor="#0F172A", linewidth=0.3)
    ax_wd.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_wd.axvline(0.10, color=SYN_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_wd.set_title("Wasserstein Distance (Numerical)", fontsize=10, color=TEXT_COLOR)
    ax_wd.grid(axis="x", alpha=0.2)
    
    # JSD bar
    ax_jsd = fig.add_subplot(gs[1, 2:])
    ax_jsd.set_facecolor(CARD_COLOR)
    cols_j = sorted(jsd_scores, key=jsd_scores.get)
    vals_j = [jsd_scores[c] for c in cols_j]
    colors_j = [GOOD_COLOR if v < 0.05 else (SYN_COLOR if v < 0.1 else BAD_COLOR) for v in vals_j]
    ax_jsd.barh(cols_j, vals_j, color=colors_j, height=0.45, edgecolor="#0F172A", linewidth=0.3)
    ax_jsd.axvline(0.05, color=GOOD_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_jsd.axvline(0.10, color=SYN_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_jsd.set_title("Jensen-Shannon Divergence (Categorical)", fontsize=10, color=TEXT_COLOR)
    ax_jsd.grid(axis="x", alpha=0.2)
    
    # Density/Coverage gauges
    for k, (metric, val) in enumerate([("Density", density), ("Coverage", coverage)]):
        ax = fig.add_subplot(gs[2, k])
        ax.set_facecolor(CARD_COLOR)
        color = GOOD_COLOR if 0.8 <= val <= 1.2 else (SYN_COLOR if 0.6 <= val <= 1.4 else BAD_COLOR)
        ax.barh([0], [1.5], color="#334155", height=0.35, edgecolor="none")
        ax.barh([0], [min(val, 1.5)], color=color, height=0.35, edgecolor="none")
        ax.axvline(1.0, color=TEXT_COLOR, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlim(0, 1.5)
        ax.set_yticks([])
        ax.text(min(val, 1.5) / 2, 0, f"{val:.4f}", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
        ax.set_title(metric, fontsize=11, color=TEXT_COLOR)
        ax.grid(axis="x", alpha=0.2)
    
    # Legend
    ax_leg = fig.add_subplot(gs[2, 2:])
    ax_leg.set_facecolor(CARD_COLOR)
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    for spine in ax_leg.spines.values():
        spine.set_edgecolor("#334155")
    
    legend_items = [
        (GOOD_COLOR, "GOOD", "WD/JSD < 0.05  |  D&C ≈ 1.0"),
        (SYN_COLOR, "OK", "WD/JSD < 0.10  |  D&C ≈ 0.8"),
        (BAD_COLOR, "POOR", "WD/JSD ≥ 0.10  |  D&C < 0.6"),
        (REAL_COLOR, "Real", "Real data distribution"),
        (SYN_COLOR, "Synth", "Synthetic data distribution"),
    ]
    for i, (c, label, desc) in enumerate(legend_items):
        y = 0.85 - i * 0.18
        ax_leg.add_patch(mpatches.FancyBboxPatch((0.03, y - 0.06), 0.08, 0.10,
                boxstyle="round,pad=0.01", facecolor=c, edgecolor="none", transform=ax_leg.transAxes))
        ax_leg.text(0.15, y, f"{label}:", fontsize=8, fontweight="bold", color=c, transform=ax_leg.transAxes, va="center")
        ax_leg.text(0.30, y, desc, fontsize=7.5, color=MUTED_COLOR, transform=ax_leg.transAxes, va="center")
    
    ax_leg.set_title("Metric Colour Guide", fontsize=9, color=TEXT_COLOR)
    
    fig.suptitle("Synthetic Data Quality Dashboard (Cond1)", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.01)
    plt.savefig(os.path.join(OUT_DIR, "summary_dashboard.png"), dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print("  Saved summary_dashboard.png")



def main() -> None:
    print("Loading condition-1 synthetic and real data...")
    X_num_real = np.load(os.path.join(REAL_DATA_PATH, "X_num_train.npy"), allow_pickle=True).astype(np.float32)
    X_cat_real = np.load(os.path.join(REAL_DATA_PATH, "X_cat_train.npy"), allow_pickle=True)
    y_real = np.load(os.path.join(REAL_DATA_PATH, "y_train.npy"), allow_pickle=True)

    X_num_syn = np.load(os.path.join(SYNTHETIC_PATH, "X_num_synthetic.npy"), allow_pickle=True).astype(np.float32)
    X_cat_syn = np.load(os.path.join(SYNTHETIC_PATH, "X_cat_synthetic.npy"), allow_pickle=True)
    y_syn = np.load(os.path.join(SYNTHETIC_PATH, "y_synthetic.npy"), allow_pickle=True)

    print(f"Real rows: {len(X_num_real):,}")
    print(f"Synthetic rows: {len(X_num_syn):,}")

    wasserstein_scores = {}
    for i, col in enumerate(NUM_COLS):
        r = X_num_real[:, i]
        s = X_num_syn[:, i]
        lo, hi = min(r.min(), s.min()), max(r.max(), s.max())
        if hi > lo:
            r_n = (r - lo) / (hi - lo)
            s_n = (s - lo) / (hi - lo)
        else:
            r_n, s_n = r, s
        wasserstein_scores[col] = float(wasserstein_distance(r_n, s_n))

    jsd_scores = {}
    for i, col in enumerate(CAT_COLS):
        r_col = X_cat_real[:, i].astype(str)
        s_col = X_cat_syn[:, i].astype(str)
        cats = sorted(set(r_col.tolist()) | set(s_col.tolist()))
        r_counts = {c: 0 for c in cats}
        s_counts = {c: 0 for c in cats}
        for v in r_col:
            r_counts[v] += 1
        for v in s_col:
            s_counts[v] += 1
        p = np.array([r_counts[c] for c in cats], dtype=float)
        q = np.array([s_counts[c] for c in cats], dtype=float)
        jsd_scores[col] = float(jsd(p, q))

    corr_real = np.corrcoef(X_num_real, rowvar=False)
    corr_syn = np.corrcoef(X_num_syn, rowvar=False)
    corr_l2 = float(np.linalg.norm(corr_real - corr_syn, ord="fro"))

    scaler = MinMaxScaler()
    R = scaler.fit_transform(X_num_real)
    S = scaler.transform(X_num_syn)

    rng = np.random.default_rng(EVAL_SEED)

    def _stratified_sample(features, labels, max_rows):
        if len(features) <= max_rows:
            return features
        sampled_indices = []
        classes, counts = np.unique(labels, return_counts=True)
        proportions = counts / counts.sum()
        remaining = max_rows
        for cls, proportion in zip(classes, proportions):
            cls_indices = np.flatnonzero(labels == cls)
            cls_target = max(1, int(round(max_rows * proportion)))
            cls_target = min(cls_target, len(cls_indices), remaining)
            if cls_target > 0:
                sampled_indices.extend(rng.choice(cls_indices, size=cls_target, replace=False).tolist())
                remaining -= cls_target
        if remaining > 0:
            unused = np.setdiff1d(np.arange(len(features)), np.array(sampled_indices, dtype=int), assume_unique=False)
            if len(unused) > 0:
                extra = rng.choice(unused, size=min(remaining, len(unused)), replace=False)
                sampled_indices.extend(extra.tolist())
        sampled_indices = np.array(sampled_indices[:max_rows], dtype=int)
        return features[sampled_indices]

    if len(R) > MAX_EVAL:
        R_eval = _stratified_sample(R, y_real.astype(int), MAX_EVAL)
    else:
        R_eval = R
    if len(S) > MAX_EVAL:
        S_eval = _stratified_sample(S, y_syn.astype(int), MAX_EVAL)
    else:
        S_eval = S

    RR = cdist(R_eval, R_eval, metric="euclidean")
    np.fill_diagonal(RR, np.inf)
    knn_radii_real = np.partition(RR, K_DENSITY, axis=1)[:, K_DENSITY - 1]

    SR = cdist(S_eval, R_eval, metric="euclidean")

    coverage_hits = np.any(SR.T < knn_radii_real[:, None], axis=1)
    coverage = float(coverage_hits.mean())

    density_vals = []
    for s_idx in range(len(S_eval)):
        count = np.sum(SR[s_idx] < knn_radii_real)
        density_vals.append(count / K_DENSITY)
    density = float(np.mean(density_vals))

    lower_bounds = np.quantile(X_num_real, LOWER_Q, axis=0)
    upper_bounds = np.quantile(X_num_real, UPPER_Q, axis=0)

    under = np.maximum(lower_bounds.reshape(1, -1) - X_num_syn, 0.0)
    over = np.maximum(X_num_syn - upper_bounds.reshape(1, -1), 0.0)
    violation = under + over
    constraint_violation_rate = float((violation > 0).any(axis=1).mean())
    constraint_violation_magnitude = float(violation.mean())

    metrics = {
        "condition": "cond1_constraint_regularized",
        "real_data_path": REAL_DATA_PATH,
        "synthetic_data_path": SYNTHETIC_PATH,
        "n_real": int(len(X_num_real)),
        "n_synthetic": int(len(X_num_syn)),
        "mean_wasserstein_distance": float(np.mean(list(wasserstein_scores.values()))),
        "mean_jsd": float(np.mean(list(jsd_scores.values()))),
        "correlation_l2_frobenius": corr_l2,
        "density_k5": density,
        "coverage_k5": coverage,
        "constraint_quantiles": {"lower": LOWER_Q, "upper": UPPER_Q},
        "constraint_violation_rate": constraint_violation_rate,
        "constraint_violation_magnitude": constraint_violation_magnitude,
        "wasserstein_per_feature": wasserstein_scores,
        "jsd_per_feature": jsd_scores,
    }

    # Generate visualizations
    generate_visualizations(X_num_real, X_cat_real, X_num_syn, X_cat_syn,
                          wasserstein_scores, jsd_scores, corr_l2, density, coverage,
                          NUM_COLS, CAT_COLS, OUT_DIR)

    metrics_path = os.path.join(OUT_DIR, "metrics_cond1.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_path = os.path.join(OUT_DIR, "summary_cond1.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EVALUATION SUMMARY (COND1)\n")
        f.write("=" * 62 + "\n")
        f.write(f"Mean Wasserstein Distance      : {metrics['mean_wasserstein_distance']:.6f}\n")
        f.write(f"Mean Jensen-Shannon Divergence: {metrics['mean_jsd']:.6f}\n")
        f.write(f"Correlation L2 (Frobenius)    : {metrics['correlation_l2_frobenius']:.6f}\n")
        f.write(f"Density (k=5)                 : {metrics['density_k5']:.6f}\n")
        f.write(f"Coverage (k=5)                : {metrics['coverage_k5']:.6f}\n")
        f.write(f"Constraint Violation Rate     : {metrics['constraint_violation_rate']:.6f}\n")
        f.write(f"Constraint Violation Magnitude: {metrics['constraint_violation_magnitude']:.8f}\n")

    print_summary(metrics)
    print(f"Saved condition-1 metrics to {metrics_path}")
    print(f"Saved condition-1 summary to {summary_path}")


if __name__ == "__main__":
    main()
