# #!/usr/bin/env python3
# # scripts/plot_mae_bar_log.py
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- configuration ---
# dataset = "electricity"
# metrics_csv = os.path.join("baseline_results", dataset, "metrics.csv")
# out_png = os.path.join("baseline_results", dataset, "mae_bar_log.png")

# # desired method order for consistent plotting
# method_order = ["Mean", "FFill-BFill", "Linear", "KNN_5", "DeepMVI"]

# # --- load ---
# if not os.path.exists(metrics_csv):
#     raise SystemExit(f"metrics.csv not found: {metrics_csv}")

# df = pd.read_csv(metrics_csv)
# # some CSVs use slightly different method names; attempt to normalize
# df['method'] = df['method'].astype(str).str.strip()

# # ensure we plot methods in the given order but only those present
# methods_present = [m for m in method_order if m in df['method'].values]
# if not methods_present:
#     raise SystemExit("No expected methods found in metrics.csv. Found: " + ", ".join(df['method'].unique()))

# df_plot = df.set_index('method').loc[methods_present]
# maes = df_plot['MAE'].astype(float).values

# # --- fix zeros for log scale ---
# # Replace exact zeros with a small value so log-scale can render.
# # Choose eps = 10% of smallest positive MAE if available, otherwise 1e-8
# positive = maes[maes > 0]
# if positive.size > 0:
#     eps = float(np.min(positive) * 0.1)
# else:
#     eps = 1e-8
# maes_safe = np.where(maes <= 0, eps, maes)

# # --- plotting ---
# plt.figure(figsize=(10,5))
# bars = plt.bar(methods_present, maes_safe)
# plt.yscale('log')
# plt.ylabel("MAE (log scale)")
# plt.xlabel("Method")
# plt.title(f"Baseline MAE comparison ({dataset}) — log scale")
# plt.grid(axis='y', linestyle=':', alpha=0.5)

# # annotate each bar with original (non-eps) MAE value (scientific/scaled)
# for rect, raw_val in zip(bars, maes):
#     h = rect.get_height()
#     if raw_val == 0:
#         txt = f"{eps:.1e} (replaced)"
#     else:
#         txt = f"{raw_val:.2g}"
#     plt.text(rect.get_x() + rect.get_width()/2, h*1.2, txt,
#              ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# os.makedirs(os.path.dirname(out_png), exist_ok=True)
# plt.savefig(out_png, dpi=150)
# print("Saved:", out_png)
#!/usr/bin/env python3
# scripts/plot_mae_bar_log_precise.py
#!/usr/bin/env python3


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import LogLocator, NullFormatter

# # --- CONFIG ---
# dataset = "electricity"           # change to "airq" if needed
# metrics_csv = os.path.join("baseline_results", dataset, "metrics.csv")
# out_png = os.path.join("baseline_results", dataset, "mae_bar_log_fixed_ylim.png")
# method_order = ["Mean", "FFill-BFill", "Linear", "KNN_5", "DeepMVI"]
# # ---------------

# if not os.path.exists(metrics_csv):
#     raise SystemExit(f"metrics.csv not found: {metrics_csv}")

# df = pd.read_csv(metrics_csv)
# df['method'] = df['method'].astype(str).str.strip()

# # Use requested order where available
# methods_present = [m for m in method_order if m in df['method'].values]
# if not methods_present:
#     methods_present = list(df['method'].values)

# df_plot = df.set_index('method').loc[methods_present]
# maes = df_plot['MAE'].astype(float).values

# # For plotting only: if any MAE <= 0, replace with very small epsilon (we'll annotate real values)
# positive = maes[maes > 0]
# eps = float(np.min(positive) * 0.1) if positive.size > 0 else 1e-8
# maes_safe = np.where(maes <= 0, eps, maes)

# plt.rcParams.update({
#     "figure.figsize": (10, 5.5),
#     "font.size": 14,
#     "axes.titlesize": 16,
#     "axes.labelsize": 13,
# })

# fig, ax = plt.subplots()
# bar_width = 0.65
# bars = ax.bar(methods_present, maes_safe, width=bar_width, edgecolor='none')

# # FORCE log scale and fixed limits: bottom = 1e-4, top = 1e0
# ax.set_yscale('log')
# ax.set_ylim(1e-4, 1.0)

# # Grid and ticks (decade ticks only)
# ax.yaxis.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
# ax.xaxis.grid(False)
# ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
# ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=12))
# ax.yaxis.set_minor_formatter(NullFormatter())

# ax.set_xlabel("Method")
# ax.set_ylabel("MAE (log scale)")
# ax.set_title(f"Baseline MAE comparison ({dataset}) — log scale", pad=12)

# # Annotate bars with original MAE values (not the epsilon replacement)
# for rect, raw_val in zip(bars, maes):
#     # place annotation a little above the bar (works on log scale)
#     plotted_h = rect.get_height()
#     y_pos = plotted_h * 1.6
#     if raw_val <= 0:
#         label = f"{eps:.1e} (rep)"
#     elif raw_val < 1e-3:
#         label = f"{raw_val:.1e}"
#     else:
#         label = f"{raw_val:.3g}"
#     ax.text(rect.get_x() + rect.get_width() / 2, y_pos, label,
#             ha='center', va='bottom', fontsize=11)

# plt.tight_layout()
# os.makedirs(os.path.dirname(out_png), exist_ok=True)
# plt.savefig(out_png, dpi=150)
# print("Saved:", out_png)


#!/usr/bin/env python3
"""
baseline_comparison.py

Runs baseline imputation methods (Mean, Forward-fill, Linear Interpolation, KNN)
and compares them against DeepMVI results (or ground truth) on a chosen dataset.

Usage:
  source venv/bin/activate
  python3 scripts/baseline_comparison.py --dataset electricity
  python3 scripts/baseline_comparison.py --dataset airq --frac-missing 0.05
"""

# import os
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import textwrap

# PAPER_PATH = "/mnt/data/Research Paper DS.pdf"  # optional reference


# # ---------------------------
# # Baseline imputation methods
# # ---------------------------

# def mean_impute(X_obs):
#     X = X_obs.copy().astype(float)
#     col_means = np.nanmean(X, axis=0)
#     inds = np.where(np.isnan(X))
#     X[inds] = np.take(col_means, inds[1])
#     return X

# def fill_ffill_bfill(X_obs):
#     import pandas as pd
#     X = X_obs.copy().astype(float)
#     for c in range(X.shape[1]):
#         s = pd.Series(X[:, c])
#         s = s.ffill().bfill()
#         X[:, c] = s.values
#     return X

# def linear_interpolate(X_obs):
#     import pandas as pd
#     X = X_obs.copy().astype(float)
#     for c in range(X.shape[1]):
#         s = pd.Series(X[:, c])
#         s = s.interpolate(method='linear', limit_direction='both')
#         if s.isna().any():
#             s = s.fillna(s.mean())
#         X[:, c] = s.values
#     return X

# def knn_impute(X_obs, n_neighbors=5):
#     imputer = KNNImputer(n_neighbors=n_neighbors)
#     return imputer.fit_transform(X_obs)


# # ---------------------------
# # Evaluation
# # ---------------------------

# def evaluate_imputation(X_true, X_imp, eval_mask):
#     ev = eval_mask.astype(bool)
#     y_true = X_true[ev]
#     y_imp = X_imp[ev]
#     mae = float(mean_absolute_error(y_true, y_imp))
#     rmse = float(mean_squared_error(y_true, y_imp, squared=False))
#     return mae, rmse


# # ---------------------------
# # MAIN
# # ---------------------------

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset", required=True)
#     p.add_argument("--outdir", default="baseline_results")
#     p.add_argument("--frac-missing", type=float, default=0.05)
#     p.add_argument("--knn-n", type=int, default=5)
#     p.add_argument("--seed", type=int, default=0)
#     args = p.parse_args()

#     root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     data_x = os.path.join(root, "data", args.dataset, "X.npy")
#     data_a = os.path.join(root, "data", args.dataset, "A.npy")

#     if not os.path.exists(data_x):
#         raise SystemExit(f"ERROR: data file not found at {data_x}")

#     # Load full X (ground truth)
#     X_true = np.load(data_x).astype(float)

#     # Missingness mask
#     if os.path.exists(data_a):
#         A = np.load(data_a)
#         orig_missing_mask = (A == 0)
#     else:
#         orig_missing_mask = np.isnan(X_true)

#     rng = np.random.default_rng(args.seed)

#     # Create synthetic missing if none exist
#     if orig_missing_mask.sum() == 0:
#         print("No missing values — creating synthetic evaluation mask.")
#         eval_mask = (rng.random(X_true.shape) < args.frac_missing)
#         X_obs = X_true.copy()
#         X_obs[eval_mask] = np.nan
#     else:
#         print("Using existing missingness mask for evaluation.")
#         eval_mask = orig_missing_mask
#         X_obs = X_true.copy()

#     # Prepare output folder
#     out_base = os.path.join(root, args.outdir, args.dataset)
#     os.makedirs(out_base, exist_ok=True)

#     # Save observed & true
#     np.save(os.path.join(out_base, "X_observed.npy"), X_obs)
#     np.save(os.path.join(out_base, "X_true.npy"), X_true)
#     np.save(os.path.join(out_base, "eval_mask.npy"), eval_mask.astype(int))

#     # Run baselines
#     results = []
#     print("\nRunning baselines...")

#     print(" - Mean")
#     X_mean = mean_impute(X_obs)
#     np.save(os.path.join(out_base, "mean_imputed.npy"), X_mean)
#     results.append(("Mean", *evaluate_imputation(X_true, X_mean, eval_mask)))

#     print(" - Forward + backward fill")
#     X_ff = fill_ffill_bfill(X_obs)
#     np.save(os.path.join(out_base, "ffill_imputed.npy"), X_ff)
#     results.append(("FFill-BFill", *evaluate_imputation(X_true, X_ff, eval_mask)))

#     print(" - Linear interpolation")
#     X_lin = linear_interpolate(X_obs)
#     np.save(os.path.join(out_base, "linear_imputed.npy"), X_lin)
#     results.append(("Linear", *evaluate_imputation(X_true, X_lin, eval_mask)))

#     print(f" - KNN (k={args.knn_n})")
#     X_knn = knn_impute(X_obs, args.knn_n)
#     np.save(os.path.join(out_base, f"knn_{args.knn_n}_imputed.npy"), X_knn)
#     results.append((f"KNN_{args.knn_n}", *evaluate_imputation(X_true, X_knn, eval_mask)))

#     # DeepMVI result if present
#     deepmvi_path = os.path.join(root, "output_deepmvi", args.dataset, "imputed.npy")
#     if os.path.exists(deepmvi_path):
#         print(" - DeepMVI (existing imputation)")
#         X_dm = np.load(deepmvi_path)
#         np.save(os.path.join(out_base, "deepmvi_imputed.npy"), X_dm)
#         results.append(("DeepMVI", *evaluate_imputation(X_true, X_dm, eval_mask)))

#     # Create metrics CSV
#     df = pd.DataFrame(results, columns=["method", "MAE", "RMSE"])
#     df.to_csv(os.path.join(out_base, "metrics.csv"), index=False)
#     print("\nSaved metrics.csv")

#     # ------------------------------
#     # (1) LINEAR SCALE PLOT
#     # ------------------------------
#     plt.figure(figsize=(6, 4))
#     plt.bar(df["method"], df["MAE"])
#     plt.ylabel("MAE")
#     plt.title(f"Baseline MAE comparison ({args.dataset})")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_base, "mae_bar.png"))
#     plt.close()

#     # ------------------------------
#     # (2) LOG-SCALE PLOT EXACTLY LIKE YOUR REFERENCE
#     # ------------------------------

#     methods = df["method"].tolist()
#     maes = df["MAE"].astype(float).to_numpy()

#     # replace zeros with epsilon for log plot
#     positive_maes = maes[maes > 0]
#     eps = max(min(positive_maes) * 0.1, 1e-12) if positive_maes.size > 0 else 1e-8
#     maes_plot = np.where(maes <= 0, eps, maes)

#     fig, ax = plt.subplots(figsize=(11, 6))  # widescreen like your sample

#     bars = ax.bar(methods, maes_plot, width=0.65)

#     # log axis + fixed limits
#     ax.set_yscale("log")
#     ax.set_ylim(1e-4, 1e0)

#     # ticks, grid
#     ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
#     ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
#     ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#     ax.grid(axis='y', which='both', linestyle=':', alpha=0.6)

#     ax.set_xlabel("Method")
#     ax.set_ylabel("MAE (log scale)")
#     ax.set_title(f"Baseline MAE comparison ({args.dataset}) — log scale", pad=12)

#     # labels above bars
#     for rect, raw_val in zip(bars, maes):
#         x = rect.get_x() + rect.get_width() / 2
#         h = rect.get_height()
#         y = h * 1.5
#         if raw_val < 1e-3:
#             label = f"{raw_val:.1e}"
#         else:
#             label = f"{raw_val:.3g}"
#         ax.text(x, y, label, ha="center", va="bottom", fontsize=10)

#     plt.tight_layout()
#     plt.savefig(os.path.join(out_base, "mae_bar_log.png"), dpi=150)
#     plt.close()

#     print("Saved mae_bar_log.png")

#     # ---------------------------
#     # README
#     # ---------------------------
#     readme = f"""
# Baseline comparison for dataset: {args.dataset}

# Files:
#  - X_observed.npy
#  - X_true.npy
#  - eval_mask.npy
#  - *_imputed.npy
#  - metrics.csv
#  - mae_bar.png
#  - mae_bar_log.png

# Reference: {PAPER_PATH}
# """
#     open(os.path.join(out_base, "README.txt"), "w").write(textwrap.dedent(readme))
#     print("\nDone.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
baseline_comparison.py

Runs baseline imputation methods (Mean, Forward-fill, Linear Interpolation, KNN)
and compares them against DeepMVI results (or ground truth) on a chosen dataset.

Usage:
  source venv/bin/activate
  python3 scripts/baseline_comparison.py --dataset electricity
  python3 scripts/baseline_comparison.py --dataset airq --frac-missing 0.05

Outputs (for dataset "NAME"):
  baseline_results/NAME/
    metrics.csv                # MAE/RMSE per baseline
    mae_bar.png                # MAE bar plot (linear y)
    mae_bar_log.png            # MAE bar plot (log y) — configured per request
    <baseline>_imputed.npy     # imputed arrays for each baseline

Notes:
- If the dataset has no NaNs, the script will create a synthetic random mask (fraction controlled by --frac-missing).
- If you already have data/<dataset>/A.npy (mask), it will be used to determine missing entries for evaluation.
- Script expects canonical X at: data/<dataset>/X.npy
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import textwrap

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

PAPER_PATH = "/mnt/data/Research Paper DS.pdf"  # reference copy you uploaded (optional)


def mean_impute(X_obs):
    X = X_obs.copy().astype(float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(col_means, inds[1])
    return X


def fill_ffill_bfill(X_obs):
    import pandas as pd
    X = X_obs.copy().astype(float)
    for c in range(X.shape[1]):
        s = pd.Series(X[:, c])
        s = s.ffill().bfill()
        # final fallback: fill remaining with col mean (if still NaN)
        if s.isna().any():
            s = s.fillna(s.mean())
        X[:, c] = s.values
    return X


def linear_interpolate(X_obs):
    import pandas as pd
    X = X_obs.copy().astype(float)
    for c in range(X.shape[1]):
        s = pd.Series(X[:, c])
        s = s.interpolate(method='linear', limit_direction='both')
        if s.isna().any():
            s = s.fillna(s.mean())
        X[:, c] = s.values
    return X


def knn_impute(X_obs, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(X_obs)


def evaluate_imputation(X_true, X_imp, eval_mask):
    """
    eval_mask: boolean array-like (True where we evaluate / where values were missing)
    Returns (mae, rmse)
    """
    ev = np.asarray(eval_mask).astype(bool)
    # flatten the mask and the arrays to select only evaluated points
    y_true = X_true[ev]
    y_imp = X_imp[ev]
    mae = float(mean_absolute_error(y_true, y_imp))
    # note: sklearn mean_squared_error accepts squared=False to return RMSE
    rmse = float(mean_squared_error(y_true, y_imp, squared=False))
    return mae, rmse


def save_linear_mae_bar(df, out_base, dataset):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["method"], df["MAE"], width=0.6)
    ax.set_ylabel("MAE")
    ax.set_title(f"Baseline MAE comparison ({dataset})")
    plt.tight_layout()
    figpath = os.path.join(out_base, "mae_bar.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    return figpath


def save_log_mae_bar(df, out_base, dataset):
    # ensure no zeros (log scale cannot plot zeros). Replace exact zeros with tiny epsilon
    eps = 1e-12
    df_plot = df.copy()
    df_plot["MAE"] = df_plot["MAE"].replace(0.0, eps)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(df_plot["method"], df_plot["MAE"], width=0.6)

    # Log scale on y-axis
    ax.set_yscale("log")

    # Set the exact y-limits so top is 10^0 and bottom is 10^-4
    ax.set_ylim(1e-4, 1e0)

    # Put log ticks at decades 1e-4,1e-3,1e-2,1e-1,1e0
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.yaxis.set_major_formatter(mticker.LogFormatter(base=10.0, labelOnlyBase=False))

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("MAE (log scale)", fontsize=12)
    ax.set_title(f"Baseline MAE comparison ({dataset}) — log scale", fontsize=14)

    # Add annotations (scientific notation)
    for rect, val in zip(bars, df_plot["MAE"]):
        h = rect.get_height()
        ax.annotate(f"{val:.1e}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    figpath = os.path.join(out_base, "mae_bar_log.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    return figpath


def main():
    parser = argparse.ArgumentParser(description="Run baseline imputations and compare to DeepMVI")
    parser.add_argument("--dataset", required=True, help="dataset folder under data/ (e.g. electricity, airq)")
    parser.add_argument("--outdir", default="baseline_results", help="base output folder")
    parser.add_argument("--frac-missing", type=float, default=0.05, help="if dataset has no NaNs, create this fraction of random missing entries")
    parser.add_argument("--knn-n", type=int, default=5, help="n_neighbors for KNNImputer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Paths relative to project root (one level up from scripts/)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_x = os.path.join(root, "data", args.dataset, "X.npy")
    data_a = os.path.join(root, "data", args.dataset, "A.npy")  # optional availability mask

    if not os.path.exists(data_x):
        raise SystemExit(f"ERROR: data file not found at {data_x}")

    # Load true data (may contain NaNs)
    try:
        X_true = np.load(data_x)
    except Exception as e:
        raise SystemExit(f"Error loading X.npy: Failed to load {data_x}: {repr(e)}")
    X_true = X_true.astype(float)

    # Load mask if available or infer from NaNs
    if os.path.exists(data_a):
        try:
            A = np.load(data_a)
            # A : 1 observed, 0 missing (we use orig_missing_mask True where missing)
            orig_missing_mask = (A == 0)
        except Exception as e:
            raise SystemExit(f"Error loading A.npy: Failed to load {data_a}: {repr(e)}")
    else:
        orig_missing_mask = np.isnan(X_true)

    # If no missing in original, create a synthetic missing mask for evaluation
    rng = np.random.default_rng(args.seed)
    if orig_missing_mask.sum() == 0:
        print("No original missing values detected. Creating synthetic missing mask (random).")
        mask = rng.random(X_true.shape) < args.frac_missing
        X_obs = X_true.copy()
        X_obs[mask] = np.nan
        eval_mask = mask
    else:
        print("Original missing values detected. Using those for evaluation.")
        X_obs = X_true.copy()
        eval_mask = orig_missing_mask

    # Prepare outputs
    out_base = os.path.join(root, args.outdir, args.dataset)
    os.makedirs(out_base, exist_ok=True)

    # Save observed/true/eval mask for reproducibility
    np.save(os.path.join(out_base, "X_observed.npy"), X_obs)
    np.save(os.path.join(out_base, "X_true.npy"), X_true)
    np.save(os.path.join(out_base, "eval_mask.npy"), eval_mask.astype(int))

    print("Running baselines on dataset:", args.dataset)
    results = []

    # Baseline: Mean per column
    print(" - Mean imputation")
    X_mean = mean_impute(X_obs)
    np.save(os.path.join(out_base, "mean_imputed.npy"), X_mean)
    mae, rmse = evaluate_imputation(X_true, X_mean, eval_mask)
    results.append(("Mean", mae, rmse))

    # Baseline: Forward then backward fill
    print(" - Forward/Backward fill")
    X_ff = fill_ffill_bfill(X_obs)
    np.save(os.path.join(out_base, "ffill_imputed.npy"), X_ff)
    mae, rmse = evaluate_imputation(X_true, X_ff, eval_mask)
    results.append(("FFill-BFill", mae, rmse))

    # Baseline: Linear interpolation
    print(" - Linear interpolation")
    X_lin = linear_interpolate(X_obs)
    np.save(os.path.join(out_base, "linear_imputed.npy"), X_lin)
    mae, rmse = evaluate_imputation(X_true, X_lin, eval_mask)
    results.append(("Linear", mae, rmse))

    # Baseline: KNN imputer
    print(f" - KNN imputer (n={args.knn_n})")
    X_knn = knn_impute(X_obs, n_neighbors=args.knn_n)
    knn_name = f"knn_{args.knn_n}_imputed.npy"
    np.save(os.path.join(out_base, knn_name), X_knn)
    mae, rmse = evaluate_imputation(X_true, X_knn, eval_mask)
    results.append((f"KNN_{args.knn_n}", mae, rmse))

    # Evaluate DeepMVI if available
    deepmvi_path = os.path.join(root, "output_deepmvi", args.dataset, "imputed.npy")
    if os.path.exists(deepmvi_path):
        print(" - Found DeepMVI imputed result; evaluating")
        try:
            X_dm = np.load(deepmvi_path)
            np.save(os.path.join(out_base, "deepmvi_imputed.npy"), X_dm)
            mae, rmse = evaluate_imputation(X_true, X_dm, eval_mask)
            results.append(("DeepMVI", mae, rmse))
        except Exception as e:
            print("   Failed to load DeepMVI imputed array:", repr(e))
    else:
        print(" - No DeepMVI result found at", deepmvi_path)

    # Save metrics CSV
    df = pd.DataFrame(results, columns=["method", "MAE", "RMSE"])
    csv_path = os.path.join(out_base, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print("Saved metrics to", csv_path)

    # Save linear MAE bar
    try:
        figpath_lin = save_linear_mae_bar(df, out_base, args.dataset)
        print("Saved MAE bar to", figpath_lin)
    except Exception as e:
        print("Failed to save linear MAE bar:", repr(e))

    # Save log MAE bar (the one you requested style for)
    try:
        figpath_log = save_log_mae_bar(df, out_base, args.dataset)
        print("Saved MAE bar (log) to", figpath_log)
    except Exception as e:
        print("Failed to save log MAE bar:", repr(e))

    # Create a README for this baseline folder
    readme = textwrap.dedent("""\
    Baseline comparison for dataset: {dataset}

    Files in this folder:
     - X_observed.npy (dataset with NaNs used as input)
     - X_true.npy (original full data)
     - eval_mask.npy (1 where we evaluate; 0 otherwise)
     - *_imputed.npy (imputed arrays for each baseline)
     - metrics.csv (MAE and RMSE per method)
     - mae_bar.png (bar plot of MAE)
     - mae_bar_log.png (bar plot of MAE on log scale)

    Reference paper used: {paper}
    """).format(dataset=args.dataset, paper=PAPER_PATH)

    open(os.path.join(out_base, "README.txt"), "w").write(readme)
    print("Baseline results saved to", out_base)
    print("Done.")


if __name__ == "__main__":
    main()
