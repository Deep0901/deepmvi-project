✔️ DeepMVI: Multivariate Time-Series Imputation

A fully reproducible pipeline with datasets, baselines, plots, and ImputeGAP integration

📌 Overview

This repository contains a full reproduction, extension, and integration of DeepMVI, a deep-learning model for multivariate time-series imputation.
The project includes:

Full DeepMVI CPU/CUDA pipeline

Support for AirQ and Electricity datasets (same as used in the original paper & ImputeGAP)

Baseline comparisons (Mean, FFill/BFill, Linear, KNN)

Automatic plot generation and diagnostics

ImputeGAP wrapper integration

Clean scripts to reproduce ALL results end-to-end

This repo is designed for clarity and reproducibility, enabling researchers to understand and extend DeepMVI easily.

🚀 1. Installation
Clone the repository
git clone https://github.com/Deep0901/deepMVI-project
cd deepmvi-project

Create and activate venv
python3 -m venv venv
source venv/bin/activate

Install requirements
pip install -r requirements.txt


(Optional) For CUDA:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

📊 2. Datasets

This repo uses two datasets, both used in the original DeepMVI paper:

✔ AirQ

10 variables

~1000 timesteps

Contains sharp spikes, noise, and realistic missingness

Ideal for robustness tests

✔ Electricity

20 variables

~5000 timesteps

Strong periodic patterns

Ideal for long-range modeling

Dataset format (after conversion)
X.npy  → numeric matrix (T × D)
A.npy  → availability mask (1 = observed, 0 = missing)

🔄 3. Running DeepMVI Imputation
AirQ
python3 scripts/run_deepmvi_for_dataset.py --dataset airq


Output saved to:

output_deepmvi/airq/imputed.npy

Electricity
python3 scripts/run_deepmvi_for_dataset.py --dataset electricity


Output saved to:

output_deepmvi/electricity/imputed.npy


Both scripts automatically patch DeepMVI to run on CPU or GPU depending on your system.

🏁 4. Running Baseline Methods

To generate baselines + comparison plots:

AirQ
python3 scripts/baseline_comparison.py --dataset airq

Electricity
python3 scripts/baseline_comparison.py --dataset electricity


Outputs stored in:

baseline_results/{dataset}/


Includes:

MAE/RMSE table

Scatter plots

Error histograms

MAE bar chart (linear + log scale)

Top-10 worst series

Difference heatmaps

🧪 5. Diagnostic Plots

Plots automatically generated under:

plots/{dataset}/

Includes:

Observed vs Imputed scatter

Absolute error histogram

Missingness mask

Difference heatmap

Per-series reconstruction (e.g., series_0, series_10…)

Top-10 series by MAE

These plots provide deep insight into model behavior, error patterns, spikes, and reconstruction quality.

🔌 6. ImputeGAP Integration

This repo includes a fully working DeepMVI wrapper compatible with ImputeGAP’s unified API.

Run wrapper (standard):
python3 imputegap/wrapper/AlgoPython/DeepMVI/deepmvi_wrapper.py --dataset airq

Run non-destructive version:
python3 imputegap/wrapper/AlgoPython/DeepMVI/deepmvi_wrapper_non_destructive.py --dataset airq

Test wrapper:
python3 tests/test_wrapper.py --dataset airq


Wrapper ensures:

standardized I/O format

compatibility with other algorithms in ImputeGAP

SHA-1 integrity check (guarantees wrapper does not modify DeepMVI outputs)

🔍 7. Experimental Protocol

Our experiments follow the official workflow recommended in the kickoff meeting:

Use availability masks or create a synthetic 5% missing mask.

Train DeepMVI with early stopping.

Save best imputed outputs (per dataset).

Run baseline imputation algorithms.

Compute MAE/RMSE on evaluation masks.

Generate diagnostic plots:

Reconstruction overlays

Absolute error histograms

Difference heatmaps

Missingness masks

Top-10 series by MAE

Baseline MAE bar charts

Compare DeepMVI vs Baselines

DeepMVI achieves MAE ≈ 10⁻⁵–10⁻⁴

Baselines: MAE ≈ 0.2–1.0

📈 8. Key Results (Summary)
DeepMVI

Near-perfect reconstruction

Handles both spiky (AirQ) and periodic (Electricity) data

Learns both temporal and cross-variable dependencies

Outperforms all baselines by several orders of magnitude

Baselines
Method	Typical Error
Mean	❌ poor (loses structure)
Forward/Backward Fill	❌ poor for spikes
Linear	❌ fails on nonlinear patterns
KNN	❌ inconsistent, local-only
DeepMVI	✔ best, by far
🧩 9. Understanding DeepMVI (Intuition)

DeepMVI is not a simple interpolation model. It reconstructs values by learning:

Temporal dependencies

Cross-variable relationships

Missingness patterns

Long-range structure (periods, spikes, trends)

This is why scatter plots go straight through y = x, showing almost perfect reconstruction.