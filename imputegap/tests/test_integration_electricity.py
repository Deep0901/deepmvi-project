import os, subprocess, glob, numpy as np, pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RUN_SCRIPT = os.path.join(ROOT, 'scripts', 'run_deepmvi_for_dataset.py')
PLOT_SCRIPT = os.path.join(ROOT, 'scripts', 'generate_plots_for_dataset.py')
OUT_DIR = os.path.join(ROOT, 'output_deepmvi', 'electricity')
PLOTS_DIR = os.path.join(ROOT, 'plots', 'electricity')
DATA_GT = os.path.join(ROOT, 'data', 'electricity', 'X.npy')  # common GT candidate

def run(cmd):
    proc = subprocess.run(cmd, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    assert proc.returncode == 0, f"Command failed: {cmd}\n{proc.stdout}"

def test_electricity_pipeline():
    # 1. run imputer
    run(f'python "{RUN_SCRIPT}" --dataset electricity')

    # 2. check imputed.npy
    imp_path = os.path.join(OUT_DIR, 'imputed.npy')
    assert os.path.exists(imp_path), f"Missing imputed: {imp_path}"
    imp=np.load(imp_path)
    assert imp.ndim == 2
    assert not np.all(np.isnan(imp))

    # 3. generate plots
    if os.path.exists(PLOT_SCRIPT):
        run(f'python "{PLOT_SCRIPT}" --dataset electricity')

    # 4. check plots
    assert os.path.isdir(PLOTS_DIR)
    assert len(glob.glob(os.path.join(PLOTS_DIR, '*'))) > 0, "No plots found"

    # 5. optional metrics if GT exists
    if os.path.exists(DATA_GT):
        gt = np.load(DATA_GT)
        if imp.shape == gt.shape:
            mask = ~np.isnan(gt)
            diff=(imp-gt)[mask]
            mae=float(np.mean(np.abs(diff)))
            rmse=float((np.mean(diff**2))**0.5)
            assert np.isfinite(mae)
            assert np.isfinite(rmse)
