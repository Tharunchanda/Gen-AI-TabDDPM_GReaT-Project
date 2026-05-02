import os
import json
import numpy as np
import pandas as pd
import torch

import lib
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from scripts.utils_train import get_model, make_dataset


# Condition 1 novelty (constraint-regularized) paths
REAL_DATA_PATH = "data/churn"
PARENT_DIR = "exp/churn/check_constraint"
MODEL_PATH = "exp/churn/check_constraint/model.pt"
MODEL_EMA_PATH = "exp/churn/check_constraint/model_ema.pt"
OUT_DIR = "exp/churn/synthetic_cond1"

NUM_SAMPLES = 115640
BATCH_SIZE = 10000
SEED = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "mlp"
MODEL_PARAMS = {
    "num_classes": 2,
    "is_y_cond": True,
    "rtdl_params": {
        "d_layers": [256, 256, 256],
        "dropout": 0.1,
    },
}

DIFFUSION_PARAMS = {
    "num_timesteps": 1000,
    "gaussian_loss_type": "mse",
    "scheduler": "cosine",
}

T_DICT = {
    "seed": 0,
    "normalization": "quantile",
    "num_nan_policy": None,
    "cat_nan_policy": None,
    "cat_min_frequency": None,
    "cat_encoding": None,
    "y_policy": "default",
}

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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset and transforms...")
    T = lib.Transformations(**T_DICT)
    D = make_dataset(
        REAL_DATA_PATH,
        T,
        num_classes=MODEL_PARAMS["num_classes"],
        is_y_cond=MODEL_PARAMS["is_y_cond"],
        change_val=False,
    )

    K = np.array(D.get_category_sizes("train"))
    if len(K) == 0 or T_DICT["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_numerical_features = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = int(np.sum(K) + num_numerical_features)
    MODEL_PARAMS["d_in"] = d_in

    print(f"K: {K}")
    print(f"d_in: {d_in}")
    print(f"num_numerical_features: {num_numerical_features}")

    model = get_model(
        MODEL_TYPE,
        MODEL_PARAMS,
        num_numerical_features,
        category_sizes=D.get_category_sizes("train"),
    )

    model_path = MODEL_EMA_PATH if os.path.exists(MODEL_EMA_PATH) else MODEL_PATH
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {model_path}")

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        num_timesteps=DIFFUSION_PARAMS["num_timesteps"],
        gaussian_loss_type=DIFFUSION_PARAMS["gaussian_loss_type"],
        scheduler=DIFFUSION_PARAMS["scheduler"],
        device=DEVICE,
    )
    diffusion.to(DEVICE)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(
        torch.from_numpy(D.y["train"]), return_counts=True
    )

    print(f"Generating {NUM_SAMPLES} synthetic samples for condition-1...")
    x_gen, y_gen = diffusion.sample_all(
        NUM_SAMPLES,
        BATCH_SIZE,
        empirical_class_dist.float(),
        ddim=False,
    )

    X_gen = x_gen.numpy()
    y_gen = y_gen.numpy()

    X_num_unnorm = X_gen[:, :num_numerical_features]
    X_cat_unnorm = X_gen[:, num_numerical_features:] if num_numerical_features < X_gen.shape[1] else None

    np.save(os.path.join(OUT_DIR, "X_num_unnorm.npy"), X_num_unnorm)
    if X_cat_unnorm is not None:
        np.save(os.path.join(OUT_DIR, "X_cat_unnorm.npy"), X_cat_unnorm)

    X_num = D.num_transform.inverse_transform(X_num_unnorm)[:, :num_numerical_features]

    X_num_real = np.load(os.path.join(REAL_DATA_PATH, "X_num_train.npy"), allow_pickle=True)
    disc_cols = []
    for col in range(X_num_real.shape[1]):
        uniq_vals = np.unique(X_num_real[:, col])
        if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
            disc_cols.append(col)
    if disc_cols:
        X_num = lib.round_columns(X_num_real, X_num, disc_cols)

    X_cat = None
    if X_cat_unnorm is not None:
        X_cat = D.cat_transform.inverse_transform(X_cat_unnorm)

    np.save(os.path.join(OUT_DIR, "X_num_synthetic.npy"), X_num)
    np.save(os.path.join(OUT_DIR, "y_synthetic.npy"), y_gen)
    if X_cat is not None:
        np.save(os.path.join(OUT_DIR, "X_cat_synthetic.npy"), X_cat)

    df_num = pd.DataFrame(X_num, columns=NUM_COLS)
    df_cat = pd.DataFrame(X_cat, columns=CAT_COLS) if X_cat is not None else pd.DataFrame()
    df_y = pd.DataFrame({TARGET_COL: y_gen.astype(int)})
    df_synthetic = pd.concat([df_num, df_cat, df_y], axis=1)

    int_like_cols = [
        "NumOfProducts",
        "NumComplaints",
        "Number of Dependents",
        "Customer Tenure",
        "Credit History Length",
    ]
    for col in int_like_cols:
        if col in df_synthetic.columns:
            df_synthetic[col] = df_synthetic[col].round(0).astype(int)

    csv_path = os.path.join(OUT_DIR, "synthetic_data_cond1.csv")
    df_synthetic.to_csv(csv_path, index=False)

    meta = {
        "condition": "cond1_constraint_regularized",
        "real_data_path": REAL_DATA_PATH,
        "model_path": model_path,
        "out_dir": OUT_DIR,
        "num_samples": int(NUM_SAMPLES),
        "batch_size": int(BATCH_SIZE),
        "seed": int(SEED),
    }
    with open(os.path.join(OUT_DIR, "generation_meta_cond1.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved condition-1 synthetic arrays and CSV in {OUT_DIR}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
