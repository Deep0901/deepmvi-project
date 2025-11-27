#!/usr/bin/env python3
"""
Robust baseline comparison script.

Usage:
    python3 scripts/baseline_comparison.py --dataset electricity
    python3 scripts/baseline_comparison.py --dataset airq

Produces:
 - baseline_results/<dataset>/metrics.csv
 - baseline_results/<dataset>/*_imputed.npy (for each baseline)
 - baseline_results/<dataset>/mae_bar.png
 - baseline_results/<dataset>/mae_bar_log.png

Notes:
 - Expects dataset folder at data/<dataset>/ with X.npy and optionally X_true.npy, A.npy, eval_mask.npy
 - If eval_mask.npy is missing, will create a 5% random eval mask (deterministic RNG seed 0).
 - Computes MAE and RMSE only over eval_mask positions (where eval_mask==1).
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Dict

# Try to import KNNImputer; if not available, we'll skip KNN baseline
try:
    from sklearn.impute import KNNImputer
    _HAS_KNN = True
except Exception:
    _HAS_KNN = False

# For consistent plotting behavior (avoid seaborn)
plt.rcParams.update({"figure.autolayout": True})


def safe_load_npy(path: str):
    """Load a numpy .npy file safely (raise informative error)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    try:
        return np.load(path)
    except Exception as e:
        # If object-pickled arrays present, inform user to recreate dataset
        raise RuntimeError(f"Failed to load {path}: {e}")


def ensure_eval_mask(dataset_dir: str, X: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """
    Ensure dataset_dir/eval_mask.npy exists. If missing, create one by sampling `fraction` of observed points.
    Returns the mask (same shape as X, dtype=int: 1 -> evaluate).
    """
    eval_path = os.path.join(dataset_dir, "eval_mask.npy")
    if os.path.exists(eval_path):
        mask = np.load(eval_path)
        if mask.shape != X.shape:
            raise RuntimeError(f"Existing eval_mask.npy has shape {mask.shape} but expected {X.shape}")
        return mask.astype(int)
    # Create a deterministic 5% eval mask from observed values
    obs = (~np.isnan(X)).nonzero()
    total = len(obs[0])
    if total == 0:
        raise RuntimeError("No observed entries found to create eval_mask.")
    k = max(1, int(total * fraction))
    rng = np.random.RandomState(0)
    idx = rng.choice(total, k, replace=False)
    M = np.zeros_like(X, dtype=int)
    M[obs[0][idx], obs[1][idx]] = 1
    np.save(eval_path, M)
    return M


def evaluate_imputation(y_true: np.ndarray, y_imp: np.ndarray, eval_mask: np.ndarray) -> Tuple[float, float]:
    """Compute MAE and RMSE over positions where eval_mask == 1."""
    if y_true.shape != y_imp.shape:
        raise ValueError("y_true and y_imp must have same shape")
    if y_true.shape != eval_mask.shape:
        raise ValueError("eval_mask shape must match data shape")
    mask = eval_mask.astype(bool)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    y_true_v = y_true[mask]
    y_imp_v = y_imp[mask]
    # ensure numeric
    y_true_v = y_true_v.astype(float)
    y_imp_v = y_imp_v.astype(float)
    mae = float(np.mean(np.abs(y_true_v - y_imp_v)))
    rmse = float(math.sqrt(float(np.mean((y_true_v - y_imp_v) ** 2))))
    return mae, rmse


def mean_impute(X_obs: np.ndarray) -> np.ndarray:
    """Column-wise mean imputation."""
    X = X_obs.copy().astype(float)
    col_mean = np.nanmean(X, axis=0)
    # where whole column is NaN, fill with 0
    nancol = np.isnan(col_mean)
    if nancol.any():
        col_mean[nancol] = 0.0
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return X


def ffill_bfill_impute(X_obs: np.ndarray) -> np.ndarray:
    """Forward fill then backward fill per series (axis=0)."""
    X = X_obs.copy().astype(float)
    # iterate over columns (time-series)
    for c in range(X.shape[1]):
        col = X[:, c]
        # forward fill
        mask = np.isnan(col)
        if not mask.all():
            # forward
            for i in range(1, len(col)):
                if np.isnan(col[i]) and (not np.isnan(col[i-1])):
                    col[i] = col[i-1]
            # backward fill
            for i in range(len(col)-2, -1, -1):
                if np.isnan(col[i]) and (not np.isnan(col[i+1])):
                    col[i] = col[i+1]
            # still NaNs (all NaNs) -> fill 0
            col[np.isnan(col)] = 0.0
        else:
            col[:] = 0.0
        X[:, c] = col
    return X


def linear_interpolate_impute(X_obs: np.ndarray) -> np.ndarray:
    """Linear interpolation along time axis for each series."""
    X = X_obs.copy().astype(float)
    n_t, n_s = X.shape
    for c in range(n_s):
        col = X[:, c]
        nans = np.isnan(col)
        if nans.all():
            X[:, c] = 0.0
            continue
        if not nans.any():
            continue
        # indices
        idx = np.arange(n_t)
        good = ~nans
        # numpy interpolation requires at least one good value
        try:
            interp = np.interp(idx, idx[good], col[good])
            X[:, c] = interp
        except Exception:
            # fallback: forward/back fill
            tmp = col.copy()
            for i in range(1, len(tmp)):
                if np.isnan(tmp[i]) and (not np.isnan(tmp[i-1])):
                    tmp[i] = tmp[i-1]
            for i in range(len(tmp)-2, -1, -1):
                if np.isnan(tmp[i]) and (not np.isnan(tmp[i+1])):
                    tmp[i] = tmp[i+1]
            tmp[np.isnan(tmp)] = 0.0
            X[:, c] = tmp
    return X


def knn_impute(X_obs: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """KNN imputation on flattened data (samples=time, features=series)."""
    if not _HAS_KNN:
        raise RuntimeError("sklearn.impute.KNNImputer not available in this environment.")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # KNNImputer expects 2D array (n_samples, n_features) -> our X_obs already has shape (T, D)
    return imputer.fit_transform(X_obs)


def safe_save_imputed(arr: np.ndarray, out_path: str):
    """Save imputed numpy array, ensure directory exists."""
    d = os.path.dirname(out_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    np.save(out_path, arr)


def plot_mae_bars(metrics: Dict[str, Tuple[float, float]], out_dir: str):
    """
    metrics: dict method -> (MAE, RMSE)
    Save two barplots: linear scale and log scale (log defaults to small eps for zeros).
    """
    methods = list(metrics.keys())
    maes = [metrics[m][0] for m in methods]
    # replace nan with large value for sorting/plotting
    maes_for_plot = [m if (m is not None and not np.isnan(m)) else float("nan") for m in maes]

    df = pd.DataFrame({"method": methods, "MAE": maes_for_plot})
    df = df.sort_values("MAE", ascending=True)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(df)), 5))
    ax.bar(df["method"], df["MAE"])
    ax.set_ylabel("MAE")
    ax.set_title("Imputation MAE by method")
    ax.set_xticklabels(df["method"], rotation=45, ha="right")
    os.makedirs(out_dir, exist_ok=True)
    out_lin = os.path.join(out_dir, "mae_bar.png")
    fig.savefig(out_lin)
    plt.close(fig)

    # log plot: avoid log(0) by adding small epsilon if needed
    eps = 1e-12
    maes_nonan = np.array([v if (v is not None and not np.isnan(v)) else np.nan for v in df["MAE"].values], dtype=float)
    # replace zeros and nans by eps to show on log scale
    maes_for_log = np.where(np.isnan(maes_nonan) | (maes_nonan <= 0.0), eps, maes_nonan)

    fig2, ax2 = plt.subplots(figsize=(max(6, 0.6 * len(df)), 5))
    ax2.bar(df["method"], maes_for_log)
    ax2.set_yscale("log")
    ax2.set_ylabel("MAE (log scale)")
    ax2.set_title("Imputation MAE by method (log scale)")
    ax2.set_xticklabels(df["method"], rotation=45, ha="right")
    out_log = os.path.join(out_dir, "mae_bar_log.png")
    fig2.savefig(out_log)
    plt.close(fig2)
    return out_lin, out_log


def main():
    parser = argparse.ArgumentParser(description="Run baseline imputation methods and compare with DeepMVI.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/ (e.g., airq, electricity)")
    parser.add_argument("--outdir", default="baseline_results", help="Base output directory")
    parser.add_argument("--knn-n", type=int, default=5, help="n neighbors for KNN imputer")
    parser.add_argument("--eval-frac", type=float, default=0.05, help="Fraction of observed points to hold out for evaluation if eval_mask missing")
    args = parser.parse_args()

    ds_dir = os.path.join("data", args.dataset)
    if not os.path.isdir(ds_dir):
        print(f"Dataset folder not found: {ds_dir}", file=sys.stderr)
        sys.exit(2)

    # Load X (observed with NaNs) and optionally X_true
    try:
        X = safe_load_npy(os.path.join(ds_dir, "X.npy"))
    except Exception as e:
        print("Error loading X.npy:", e, file=sys.stderr)
        sys.exit(3)

    X_true_path = os.path.join(ds_dir, "X_true.npy")
    if os.path.exists(X_true_path):
        X_true = np.load(X_true_path)
    else:
        # If X_true is missing, but A.npy exists as availability mask and we have a backup true array, prefer X.npy (original)
        # We'll treat X_true as the same array if not available (so evaluation only on artificially created mask is meaningful).
        X_true = X.copy()

    # Ensure shape consistency
    if X.ndim != 2:
        raise RuntimeError("X.npy must be a 2D array (T x D).")

    # Ensure eval_mask exists
    try:
        eval_mask = ensure_eval_mask(ds_dir, X, fraction=args.eval_frac)
    except Exception as e:
        print("Failed to create/validate eval_mask:", e, file=sys.stderr)
        sys.exit(4)

    # Output directories
    out_base = os.path.join(args.outdir, args.dataset)
    os.makedirs(out_base, exist_ok=True)

    # Prepare list of methods
    results = {}
    imputed_paths = {}

    # 1) Mean imputation
    print(" - Mean imputation")
    X_mean = mean_impute(X)
    p_mean = os.path.join(out_base, "mean_imputed.npy")
    safe_save_imputed(X_mean, p_mean)
    imputed_paths["Mean"] = p_mean
    results["Mean"] = evaluate_imputation(X_true, X_mean, eval_mask)

    # 2) Forward/Backward fill
    print(" - Forward/Backward fill")
    X_ffill = ffill_bfill_impute(X)
    p_ffill = os.path.join(out_base, "ffill_imputed.npy")
    safe_save_imputed(X_ffill, p_ffill)
    imputed_paths["FFill-BFill"] = p_ffill
    results["FFill-BFill"] = evaluate_imputation(X_true, X_ffill, eval_mask)

    # 3) Linear interpolation
    print(" - Linear interpolation")
    X_linear = linear_interpolate_impute(X)
    p_linear = os.path.join(out_base, "linear_imputed.npy")
    safe_save_imputed(X_linear, p_linear)
    imputed_paths["Linear"] = p_linear
    results["Linear"] = evaluate_imputation(X_true, X_linear, eval_mask)

    # 4) KNN imputer (if available)
    if _HAS_KNN:
        try:
            print(f" - KNN imputer (n={args.knn_n})")
            X_knn = knn_impute(X, n_neighbors=args.knn_n)
            p_knn = os.path.join(out_base, f"knn_{args.knn_n}_imputed.npy")
            safe_save_imputed(X_knn, p_knn)
            imputed_paths[f"KNN_{args.knn_n}"] = p_knn
            results[f"KNN_{args.knn_n}"] = evaluate_imputation(X_true, X_knn, eval_mask)
        except Exception as e:
            print("KNN imputer failed:", e, file=sys.stderr)
    else:
        print(" - KNN imputer not available (skipping).")

    # 5) DeepMVI: check for existing imputed output produced by DeepMVI pipeline/wrapper
    deepmvi_path_candidates = [
        os.path.join("output_deepmvi", args.dataset, "imputed.npy"),
        os.path.join("output_deepmvi", "imputed.npy"),
        os.path.join("output_deepmvi", args.dataset, "imputed_from_wrapper.npy"),
    ]
    deepmvi_found = None
    for p in deepmvi_path_candidates:
        if os.path.exists(p):
            deepmvi_found = p
            break
    if deepmvi_found:
        print(" - DeepMVI imputed array found at:", deepmvi_found)
        try:
            X_deep = np.load(deepmvi_found)
            p_deep = os.path.join(out_base, "deepmvi_imputed.npy")
            safe_save_imputed(X_deep, p_deep)
            imputed_paths["DeepMVI"] = p_deep
            results["DeepMVI"] = evaluate_imputation(X_true, X_deep, eval_mask)
        except Exception as e:
            print("Failed to load DeepMVI imputed array:", e, file=sys.stderr)
            results["DeepMVI"] = (float("nan"), float("nan"))
    else:
        print(" - No DeepMVI imputed array found; skipping DeepMVI entry.")
        results["DeepMVI"] = (float("nan"), float("nan"))

    # Save metrics.csv
    rows = []
    for method, (mae, rmse) in results.items():
        rows.append({"method": method, "MAE": mae, "RMSE": rmse})
    dfm = pd.DataFrame(rows).sort_values("MAE", ascending=True)
    metrics_csv = os.path.join(out_base, "metrics.csv")
    dfm.to_csv(metrics_csv, index=False)
    print("Saved metrics to", metrics_csv)

    # Generate MAE bar plots
    try:
        plot_mae_bars(results, out_base)
        print("Saved MAE bar plots to", out_base)
    except Exception as e:
        print("Failed to generate MAE plots:", e, file=sys.stderr)

    print("Baseline results saved to", out_base)
    print("Done.")


if __name__ == "__main__":
    main()
