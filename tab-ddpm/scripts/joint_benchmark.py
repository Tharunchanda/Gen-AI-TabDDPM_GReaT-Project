#!/usr/bin/env python3
"""
Joint utility-fidelity-risk benchmark

Usage examples:
  python scripts/joint_benchmark.py \
    --real-train data/churn/train.csv \
    --real-test data/churn/test.csv \
    --synthetic exp/churn/synthetic_cond1/samples.csv \
    --label-col Churn \
    --out exp/churn/joint_benchmark_cond1.json

The script computes:
 - fidelity: per-feature Wasserstein (numerical), mean JSD (categorical), correlation L2, density, coverage
 - downstream utility: train-on-synthetic -> test-on-real AUC & F1 (and train-on-real reference)
 - privacy risk: simple NN-based membership inference AUROC (real_train vs real_test distances to synthetic)

This file is standalone and does not modify other repository files.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import jensenshannon
except Exception:
    print("scipy is required. Install with: pip install scipy")
    raise

from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


def detect_label_column(df):
    for candidate in ("churn", "Churn", "target", "label", "Exited", "exit"):
        if candidate in df.columns:
            return candidate
    return None


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def numeric_and_categorical(df, cat_cols=None, num_cols=None):
    if num_cols is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if cat_cols is None:
        cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols


def compute_wasserstein(real_num, synth_num):
    per_feature = {}
    for col in real_num.columns:
        a = real_num[col].dropna().values
        b = synth_num[col].dropna().values
        if len(a) == 0 or len(b) == 0:
            per_feature[col] = None
            continue
        per_feature[col] = float(wasserstein_distance(a, b))
    mean = float(np.nanmean([v for v in per_feature.values() if v is not None]))
    return {"per_feature": per_feature, "mean": mean}


def compute_jsd(real_cat, synth_cat):
    per_feature = {}
    for col in real_cat.columns:
        ra = real_cat[col].fillna("__nan__")
        sa = synth_cat[col].fillna("__nan__")
        rcounts = ra.value_counts(normalize=True)
        scounts = sa.value_counts(normalize=True)
        # Align indices
        idx = sorted(set(rcounts.index) | set(scounts.index))
        p = np.array([rcounts.get(i, 0.0) for i in idx], dtype=float)
        q = np.array([scounts.get(i, 0.0) for i in idx], dtype=float)
        # jensenshannon returns sqrt(JS); square to get divergence
        try:
            js = float(jensenshannon(p, q) ** 2)
        except Exception:
            js = float(0.0)
        per_feature[col] = js
    mean = float(np.nanmean([v for v in per_feature.values() if v is not None]))
    return {"per_feature": per_feature, "mean": mean}


def correlation_l2(real_num, synth_num):
    # compute Pearson correlation matrices
    r_corr = real_num.corr().fillna(0).values
    s_corr = synth_num.corr().fillna(0).values
    diff = r_corr - s_corr
    return float(np.linalg.norm(diff, ord="fro"))


def density_and_coverage(real_num, synth_num, k=5):
    # density: mean distance from synthetic points to nearest real neighbor (lower is better)
    nbrs_real = NearestNeighbors(n_neighbors=1).fit(real_num.values)
    dists, _ = nbrs_real.kneighbors(synth_num.values)
    mean_nn_dist = float(np.mean(dists[:, 0]))
    # coverage: fraction of real points that are nearest neighbor of at least one synthetic point
    nbrs_synth = NearestNeighbors(n_neighbors=1).fit(synth_num.values)
    _, idx = nbrs_synth.kneighbors(real_num.values)
    covered = len(set(idx[:, 0]))
    coverage = float(covered / float(len(real_num)))
    return {"density_mean_nn_dist": mean_nn_dist, "coverage_fraction": coverage}


def preprocess_for_classifier(df, numeric_cols, categorical_cols, label_col=None):
    # simple preprocessing: fillna, one-hot categorical, keep numeric
    dfc = df.copy()
    dfc[numeric_cols] = dfc[numeric_cols].fillna(0.0)
    dfc[categorical_cols] = dfc[categorical_cols].fillna("__nan__")
    df_enc = pd.get_dummies(dfc[categorical_cols].astype(str), drop_first=False)
    X = pd.concat([dfc[numeric_cols].reset_index(drop=True), df_enc.reset_index(drop=True)], axis=1)
    if label_col and label_col in dfc.columns:
        y = dfc[label_col].values
    else:
        y = None
    return X, y


def align_columns(X_train, X_test):
    # ensure same columns
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    for c in X_test.columns:
        if c not in X_train.columns:
            X_train[c] = 0
    X_train = X_train[X_test.columns]
    return X_train, X_test


def downstream_metrics(real_train, real_test, synth, numeric_cols, categorical_cols, label_col):
    Xs, ys = preprocess_for_classifier(synth, numeric_cols, categorical_cols, label_col)
    Xr_train, yr_train = preprocess_for_classifier(real_train, numeric_cols, categorical_cols, label_col)
    Xr_test, yr_test = preprocess_for_classifier(real_test, numeric_cols, categorical_cols, label_col)
    if yr_test is None:
        raise ValueError("Label column not found in real_test")
    # align columns
    Xs, Xr_test = align_columns(Xs, Xr_test)
    Xr_train, Xr_test = align_columns(Xr_train, Xr_test)

    results = {}
    # train on synthetic, evaluate on real test
    if ys is not None:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(Xs.values, ys)
        probs = clf.predict_proba(Xr_test.values)[:, 1]
        preds = (probs >= 0.5).astype(int)
        results["train_on_synth_auc"] = float(roc_auc_score(yr_test, probs))
        results["train_on_synth_f1"] = float(f1_score(yr_test, preds))
    else:
        results["train_on_synth_auc"] = None
        results["train_on_synth_f1"] = None

    # reference: train on real train, eval on real test
    clf2 = RandomForestClassifier(n_estimators=100, random_state=1)
    clf2.fit(Xr_train.values, yr_train)
    probs2 = clf2.predict_proba(Xr_test.values)[:, 1]
    preds2 = (probs2 >= 0.5).astype(int)
    results["train_on_real_auc"] = float(roc_auc_score(yr_test, probs2))
    results["train_on_real_f1"] = float(f1_score(yr_test, preds2))

    return results


def membership_inference_nn(real_train, real_test, synth_num):
    # compute distance to nearest synthetic for real_train and real_test
    nbrs = NearestNeighbors(n_neighbors=1).fit(synth_num.values)
    d_train, _ = nbrs.kneighbors(real_train.values)
    d_test, _ = nbrs.kneighbors(real_test.values)
    # lower distance => more likely memorized; use negative distance as score
    y_true = np.concatenate([np.ones(len(d_train)), np.zeros(len(d_test))])
    scores = np.concatenate([-d_train.ravel(), -d_test.ravel()])
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = None
    return {"mi_nn_auroc": auroc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-train", required=True)
    p.add_argument("--real-test", required=True)
    p.add_argument("--synthetic", required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--numeric-cols", default=None,
                   help="comma-separated numeric columns (optional)")
    p.add_argument("--categorical-cols", default=None,
                   help="comma-separated categorical columns (optional)")
    p.add_argument("--out", default="joint_benchmark_results.json")
    args = p.parse_args()

    real_train = load_csv(args.real_train)
    real_test = load_csv(args.real_test)
    synth = load_csv(args.synthetic)

    label_col = args.label_col or detect_label_column(real_train) or detect_label_column(real_test)
    if label_col is None:
        print("Warning: could not auto-detect label column. Use --label-col to specify. Downstream metrics will error.")

    if args.numeric_cols:
        numeric_cols = [c.strip() for c in args.numeric_cols.split(",")]
    else:
        numeric_cols, _ = numeric_and_categorical(real_train)

    if args.categorical_cols:
        categorical_cols = [c.strip() for c in args.categorical_cols.split(",")]
    else:
        # treat non-numeric (except label) as categorical
        _, cat_cols = numeric_and_categorical(real_train)
        categorical_cols = [c for c in cat_cols if c != label_col]

    # select columns present in dataframes
    numeric_cols = [c for c in numeric_cols if c in real_train.columns and c in synth.columns]
    categorical_cols = [c for c in categorical_cols if c in real_train.columns and c in synth.columns]

    # compute fidelity
    real_num = real_train[numeric_cols]
    synth_num = synth[numeric_cols]
    real_cat = real_train[categorical_cols]
    synth_cat = synth[categorical_cols]

    w = compute_wasserstein(real_num, synth_num)
    jsd = compute_jsd(real_cat, synth_cat) if len(categorical_cols) > 0 else {"per_feature": {}, "mean": None}
    corr_l2 = correlation_l2(real_num, synth_num)
    dens_cov = density_and_coverage(real_num, synth_num, k=5)

    downstream = downstream_metrics(real_train, real_test, synth, numeric_cols, categorical_cols, label_col)
    privacy = membership_inference_nn(real_train[numeric_cols], real_test[numeric_cols], synth_num)

    out = {
        "fidelity": {"wasserstein": w, "jsd": jsd, "correlation_l2": corr_l2, "density_coverage": dens_cov},
        "downstream": downstream,
        "privacy": privacy,
        "meta": {"real_train_rows": len(real_train), "real_test_rows": len(real_test), "synth_rows": len(synth)}
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("Wrote results to", args.out)


if __name__ == "__main__":
    main()
