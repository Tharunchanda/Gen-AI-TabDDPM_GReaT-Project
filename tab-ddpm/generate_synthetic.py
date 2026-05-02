import torch
import numpy as np
import os
import sys
import json
import pandas as pd

sys.path.insert(0, '.')

import lib
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from scripts.utils_train import get_model, make_dataset

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change these as needed
# ═══════════════════════════════════════════════════════════════════════════════
REAL_DATA_PATH = "data/churn"
PARENT_DIR     = "exp/churn/check"          # where model.pt lives
MODEL_PATH     = "exp/churn/check/model.pt"
OUT_DIR        = "exp/churn/synthetic"
NUM_SAMPLES    = 5000                       # ← how many rows to generate
BATCH_SIZE     = 100
SEED           = 0
DEVICE         = torch.device("cpu")        # change to "cuda:0" if GPU

MODEL_TYPE     = "mlp"
MODEL_PARAMS = {
    "num_classes": 2,
    "is_y_cond":   True,
    "rtdl_params": {
        "d_layers": [256, 256, 256],   # match config
        "dropout":  0.1                # match config
    }
}
DIFFUSION_PARAMS = {
    "num_timesteps":      1000,
    "gaussian_loss_type": "mse",
    "scheduler":          "cosine"
}
T_DICT = {
    "seed":             0,
    "normalization":    "quantile",
    "num_nan_policy":   None,
    "cat_nan_policy":   None,
    "cat_min_frequency": None,
    "cat_encoding":     None,
    "y_policy":         "default"
}

# ── Column names (must match your preprocessing) ──────────────────────────────
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
CAT_COLS = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Customer Segment",
    "Preferred Communication Channel",
]
TARGET_COL = "Churn Flag"

# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 2. Load dataset (same as training — needed for transforms + category sizes)
print("Loading dataset and transforms...")
T = lib.Transformations(**T_DICT)
D = make_dataset(
    REAL_DATA_PATH,
    T,
    num_classes=MODEL_PARAMS["num_classes"],
    is_y_cond=MODEL_PARAMS["is_y_cond"],
    change_val=False
)

# ── 3. Build model ────────────────────────────────────────────────────────────
K = np.array(D.get_category_sizes('train'))
if len(K) == 0 or T_DICT['cat_encoding'] == 'one-hot':
    K = np.array([0])

num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
d_in = int(np.sum(K) + num_numerical_features_)
MODEL_PARAMS['d_in'] = d_in

print(f"K (cat cardinalities): {K}")
print(f"d_in: {d_in}")
print(f"num_numerical_features: {num_numerical_features_}")

model = get_model(
    MODEL_TYPE,
    MODEL_PARAMS,
    num_numerical_features_,
    category_sizes=D.get_category_sizes('train')
)

# ── 4. Load trained weights ───────────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(DEVICE)
model.eval()
print(f"\n✅ Loaded model from {MODEL_PATH}")

# ── 5. Build diffusion ────────────────────────────────────────────────────────
diffusion = GaussianMultinomialDiffusion(
    K,
    num_numerical_features = num_numerical_features_,
    denoise_fn             = model,
    num_timesteps          = DIFFUSION_PARAMS["num_timesteps"],
    gaussian_loss_type     = DIFFUSION_PARAMS["gaussian_loss_type"],
    scheduler              = DIFFUSION_PARAMS["scheduler"],
    device                 = DEVICE
)
diffusion.to(DEVICE)
diffusion.eval()

# ── 6. Compute empirical class distribution from training data ─────────────────
_, empirical_class_dist = torch.unique(
    torch.from_numpy(D.y['train']),
    return_counts=True
)
print(f"\nEmpirical class distribution: {empirical_class_dist.tolist()}")

# ── 7. Generate samples ───────────────────────────────────────────────────────
print(f"\nGenerating {NUM_SAMPLES} synthetic samples...")
x_gen, y_gen = diffusion.sample_all(
    NUM_SAMPLES,
    BATCH_SIZE,
    empirical_class_dist.float(),
    ddim=False
)

X_gen  = x_gen.numpy()
y_gen  = y_gen.numpy()

print(f"Raw generated shape: {X_gen.shape}")

# ── 8. Save unnormalized arrays (same as scripts/sample.py) ───────────────────
np.save(os.path.join(PARENT_DIR, 'X_num_unnorm'), X_gen[:, :num_numerical_features_])
if num_numerical_features_ < X_gen.shape[1]:
    np.save(os.path.join(PARENT_DIR, 'X_cat_unnorm'), X_gen[:, num_numerical_features_:])

# ── 9. Inverse transform numerical features ───────────────────────────────────
X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features_])
X_num  = X_num_[:, :num_numerical_features_]

# Round discrete/integer numerical columns (e.g. NumOfProducts, NumComplaints)
X_num_real = np.load(os.path.join(REAL_DATA_PATH, "X_num_train.npy"), allow_pickle=True)
disc_cols = []
for col in range(X_num_real.shape[1]):
    uniq_vals = np.unique(X_num_real[:, col])
    if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
        disc_cols.append(col)
print(f"Discrete numerical cols (will be rounded): {disc_cols}")
if len(disc_cols):
    X_num = lib.round_columns(X_num_real, X_num, disc_cols)

# ── 10. Inverse transform categorical features ────────────────────────────────
X_cat = None
if num_numerical_features_ < X_gen.shape[1]:
    X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features_:])

# ── 11. Save .npy files ───────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X_num_synthetic.npy"), X_num)
np.save(os.path.join(OUT_DIR, "y_synthetic.npy"),     y_gen)
if X_cat is not None:
    np.save(os.path.join(OUT_DIR, "X_cat_synthetic.npy"), X_cat)

print(f"\n✅ Saved .npy files to {OUT_DIR}/")
print(f"   X_num shape: {X_num.shape}")
if X_cat is not None:
    print(f"   X_cat shape: {X_cat.shape}")
print(f"   y     shape: {y_gen.shape}")

# ── 12. Convert to CSV with correct column names ──────────────────────────────
print("\nConverting to CSV...")

# Numerical dataframe
df_num = pd.DataFrame(X_num, columns=NUM_COLS)

# Categorical dataframe — inverse_transform gives back integer codes,
# so we need to map them back to original string labels using training data
if X_cat is not None:
    # Load original training cat data to rebuild vocab
    X_cat_train_raw = np.load(
        os.path.join(REAL_DATA_PATH, "X_cat_train.npy"),
        allow_pickle=True
    )

    # Rebuild vocab: for each column, sorted unique values → index mapping
    # (same logic as your preprocess_churn.py)
    cat_vocab = {}
    for i, col in enumerate(CAT_COLS):
        unique_vals = sorted(set(X_cat_train_raw[:, i].tolist()))
        cat_vocab[col] = {idx: val for idx, val in enumerate(unique_vals)}

    print("\nCategory vocabularies (index → label):")
    for col, vocab in cat_vocab.items():
        print(f"  {col}: {vocab}")

    # inverse_transform already returned string labels — use directly
    df_cat = pd.DataFrame(X_cat, columns=CAT_COLS)
else:
    df_cat = pd.DataFrame()

# Target
df_y = pd.DataFrame({TARGET_COL: y_gen.astype(int)})

# Combine
df_synthetic = pd.concat([df_num, df_cat, df_y], axis=1)

# ── 13. Post-processing: clip unrealistic values ──────────────────────────────
# Clip numerical columns to real data min/max to avoid out-of-range values
df_real_num = pd.DataFrame(X_num_real, columns=NUM_COLS)
for col in NUM_COLS:
    real_min = df_real_num[col].min()
    real_max = df_real_num[col].max()
    df_synthetic[col] = df_synthetic[col].clip(lower=real_min, upper=real_max)

# Round integer-like columns
int_like_cols = ["NumOfProducts", "NumComplaints", "Number of Dependents",
                 "Customer Tenure", "Credit History Length"]
for col in int_like_cols:
    if col in df_synthetic.columns:
        df_synthetic[col] = df_synthetic[col].round(0).astype(int)

# ── 14. Save CSV ──────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "synthetic_data.csv")
df_synthetic.to_csv(csv_path, index=False)

print(f"\n✅ Saved CSV to {csv_path}")
print(f"   Shape: {df_synthetic.shape}")
print(f"\nChurn distribution in synthetic data:")
print(df_synthetic[TARGET_COL].value_counts())
print(f"\nPreview (first 10 rows):")
print(df_synthetic.head(10).to_string())