import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("data/churn/churn_raw.csv")
print("Original shape:", df.shape)
print("Churn distribution:\n", df["Churn Flag"].value_counts())

# ── 2. Column definitions ─────────────────────────────────────────────────────
target_col = "Churn Flag"

drop_cols = [
    "RowNumber", "CustomerId", "Surname", "First Name",
    "Date of Birth", "Address", "Contact Information",
    "Churn Reason", "Churn Date", "Occupation"
]

num_cols = [
    "Income", "Credit Score", "Credit History Length",
    "Outstanding Loans", "Balance", "NumOfProducts",
    "NumComplaints", "Number of Dependents", "Customer Tenure",
]

cat_cols = [
    "Gender", "Marital Status",
    "Education Level", "Customer Segment",
    "Preferred Communication Channel",
]

# ── 3. Drop leakage / ID columns ─────────────────────────────────────────────
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df[target_col] = df[target_col].astype(int)
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.reset_index(drop=True)
print(f"\nUsing all {len(df)} rows | Churn rate: {df[target_col].mean():.2%}")

# ── 4. Train / Val / Test split (80 / 10 / 10) ───────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df[target_col], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 5. Fill NaNs using train statistics only ──────────────────────────────────
train_means = train_df[num_cols].mean()
train_modes = train_df[cat_cols].mode().iloc[0]

for split_df in [train_df, val_df, test_df]:
    split_df[num_cols] = split_df[num_cols].fillna(train_means)
    split_df[cat_cols] = split_df[cat_cols].fillna(train_modes)



# ── 7. Build category vocab from TRAIN only ───────────────────────────────────
cat_vocab = {}
for col in cat_cols:
    unique_vals = sorted(train_df[col].astype(str).unique())
    cat_vocab[col] = {v: idx for idx, v in enumerate(unique_vals)}

print("\nCategory cardinalities:")
for col, vocab in cat_vocab.items():
    print(f"  {col}: {len(vocab)} categories")

# ── 8. Save .npy files ────────────────────────────────────────────────────────
out_dir = "data/churn"
os.makedirs(out_dir, exist_ok=True)

def save_split(df_split, name):
    X_num = df_split[num_cols].astype(np.float32).values
    np.save(os.path.join(out_dir, f"X_num_{name}.npy"), X_num)

    X_cat = df_split[cat_cols].astype(str).values
    np.save(os.path.join(out_dir, f"X_cat_{name}.npy"), X_cat)

    y = df_split[target_col].astype(np.int64).values
    np.save(os.path.join(out_dir, f"y_{name}.npy"), y)

    print(f"  [{name}] X_num{X_num.shape} | X_cat{X_cat.shape} | y{y.shape}")

print("\nSaving .npy files...")
save_split(train_df, "train")
save_split(val_df,   "val")
save_split(test_df,  "test")

# ── 9. Save info.json ─────────────────────────────────────────────────────────
cat_cardinalities = [len(cat_vocab[col]) for col in cat_cols]

info = {
    "name":              "churn",
    "basename":          "churn",
    "split":             "custom",
    "task_type":         "binclass",
    "n_num_features":    len(num_cols),
    "n_cat_features":    len(cat_cols),
    "train_size":        len(train_df),
    "val_size":          len(val_df),
    "test_size":         len(test_df),
    "n_classes":         2,
    "cat_cardinalities": cat_cardinalities
}

with open(os.path.join(out_dir, "info.json"), "w") as f:
    json.dump(info, f, indent=2)

print("\n✅ Preprocessing complete!")
print("Files saved:", sorted(os.listdir(out_dir)))
print("\ninfo.json content:")
print(json.dumps(info, indent=2))