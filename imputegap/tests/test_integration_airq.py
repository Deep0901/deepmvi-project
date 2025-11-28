import os, subprocess, glob, numpy as np, pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUN_SCRIPT = os.path.join(ROOT, 'scripts', 'run_deepmvi_for_dataset.py')
PLOT_SCRIPT = os.path.join(ROOT, 'scripts', 'generate_plots_for_dataset.py')
OUT_DIR = os.path.join(ROOT, 'output_deepmvi', 'airq')
PLOTS_DIR = os.path.join(ROOT, 'plots', 'airq')
DATA_GT = os.path.join(ROOT, 'data', 'airq', 'X.npy')  # common GT candidate

def run_command(cmd, timeout=600):
    # run shell command, raise on non-zero exit
    proc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd}\nOutput:\\n{proc.stdout}")
    return proc.stdout

def test_run_deepmvi_and_generate_plots():
    # 1) Run the imputation pipeline
    assert os.path.exists(RUN_SCRIPT), f"Runner script not found: {RUN_SCRIPT}"
    run_command(f'python "{RUN_SCRIPT}" --dataset airq')

    # 2) Check imputed file exists
    imputed_path = os.path.join(OUT_DIR, 'imputed.npy')
    assert os.path.exists(imputed_path), f"Imputed file missing: {imputed_path}"
    imputed = np.load(imputed_path)
    assert imputed.size > 0, "Imputed array is empty"
    # basic content checks
    assert not np.all(np.isnan(imputed)), "All values are NaN in imputed output"

    # 3) Generate plots (some repos auto-make them inside runner; still run script to be safe)
    if os.path.exists(PLOT_SCRIPT):
        run_command(f'python "{PLOT_SCRIPT}" --dataset airq')

    # 4) Check there is at least one plot file
    plots = []
    if os.path.isdir(PLOTS_DIR):
        plots = glob.glob(os.path.join(PLOTS_DIR, '*'))
    assert len(plots) > 0, f"No plots found in {PLOTS_DIR}"

    # 5) If ground truth exists, compute MAE/RMSE and ensure finite
    if os.path.exists(DATA_GT):
        gt = np.load(DATA_GT)
        # attempt to align shapes
        if gt.shape == imputed.shape:
            mask = ~np.isnan(gt)
            diff = (imputed - gt)[mask]
            mae = float(np.mean(np.abs(diff)))
            rmse = float((np.mean(diff**2))**0.5)
            assert np.isfinite(mae) and np.isfinite(rmse), "MAE or RMSE not finite"
        else:
            # if shapes differ, at least assert both are 2D arrays or compatible
            assert imputed.ndim >= 2 and gt.ndim >= 2, "Unexpected data shapes"
