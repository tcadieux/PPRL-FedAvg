# CSCI 575 - Advanced Machine Learning - Final Project
**Privacy-Preserving Record Linkage: A Five-Tier Comparison Framework**



## Programming Language

Python 3.10 or newer.

The project is delivered as a set of Jupyter notebooks plus supporting `.py` modules. A small driver script (`run_all.py`) executes every notebook end-to-end.

---

## Directory Structure

`project.zip` extracts to a single directory (`Project/`) containing:

```
README.TXT              This file.
requirements.txt        Python dependencies (pip-installable).

Notebooks/              The five tiers + a final comparison notebook.
  01_tier1_hand_tuned.ipynb       Tier 1: hand-tuned scorecard (no ML).
  02_tier2_naive_bayes.ipynb      Tier 2: Gaussian Naive Bayes.
  03_tier3_centralized_ml.ipynb   Tier 3: centralized logistic regression.
  04_tier4_federated_ttp.ipynb    Tier 4: FedAvg LR on plaintext features.
  05_tier5_federated_bloom.ipynb  Tier 5: FedAvg LR on Bloom-encoded
                                  features (the privacy-preserving result).
  06_results_comparison.ipynb     Cross-tier metrics and figures.
  utils.py                        Shared helpers: paths, similarity
                                  functions, data loading, FedAvg
                                  primitives, metrics, plotting.

Data/
  output/                 Pre-computed candidate pairs (shipped):
    party_a.csv             Party A's records.
    party_b.csv             Party B's records.
    pairs.csv               Candidate pairs with ground-truth labels.

Results/                  Per-tier outputs written by the notebooks:
                            tierN_metrics.csv
                            tierN_score_distribution.png
                            tierN_example_predictions.csv
                            comparison_metrics_bar.png
                            comparison_example_pairs.csv

Report/
  report.pdf              Compiled report (this is the report to grade).
```

---

## How to Run

The dataset under `Data/output/` is shipped pre-computed, so the default run path is: install dependencies, then execute the driver.

> Tested on macOS (Darwin 25.3.0) with Python 3.12.

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `faker`, `jellyfish`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `jupyter`, and their dependencies.

### Step 2 — Execute every notebook in order

```bash
python3 run_all.py
```

This invokes `jupyter nbconvert --to notebook --execute --inplace` on each of the six notebooks (`01_*` through `06_*`) in sorted order. Total runtime is approximately 2 minutes; Tier 5's Bloom-filter encoding is the slowest stage. Outputs (CSVs and PNGs) are written into `Results/`, and the executed notebooks are saved in place so their rendered cells can be opened afterward.

### Step 3 — View the rendered notebooks

```bash
jupyter notebook Notebooks/
```

Or open any individual `.ipynb` file in VS Code / JupyterLab.
