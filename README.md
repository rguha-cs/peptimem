# PeptiMem: Uncovering Sex-Specific Memory Restoration Through Chromogranin A-Derived Peptides in Alzheimer's Disease: Computational Modeling Segment
Two-phase computational framework consisting of causal inference model and residual machine learning classfier for identifying and predicting responders to Alzheimer's Disease treatments

This project contains two analysis components for the peptide-treatment mouse study:

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

## 1. Causal Regression Modeling
This script estimates treatment effects for **CST, PST, and BOTH** compared with controls across:
- **Phase 1:** CgA-KO mice
- **Phase 2:** PS19 mice
- **Outcomes:** `SAT_END` and `MEM_END`

It computes:
- Mean treatment effects
- 95% bootstrap confidence intervals
- Permutation p-values
- Benjamini–Hochberg (BH) adjusted q-values
- Cohen’s d effect sizes
- Baseline-adjusted effects for Phase 1

It also saves:
- Tidy CSV result files
- Bootstrap and permutation distributions
- Publication-style figures (forest plots, volcano plots, p-value tables, null distributions)

## 2. Residual ML Framework
This script implements a **two-stage residual machine learning model** to predict treatment responders.

### Stage 1
A Ridge regression model is trained on **saline-treated mice** to estimate expected Week 10 memory performance.

### Stage 2
A logistic regression classifier uses baseline + early behavioral features to predict whether a mouse is a **responder** based on its residual improvement above expected performance.

It outputs:
- Cross-validation and external test metrics
- Per-mouse predictions
- Model coefficients
- Bootstrapped AUC distributions
- Figures including ROC curves, confusion matrices, and residual plots

## Input Files
Place these files in the working directory:
- `phase1.csv`
- `phase2.csv`

## Main Output Files
### Component 1
- `causal_effects_tidy.csv`
- `causal_effects_baseline_adjusted_tidy.csv`
- `C1_outputs/`

### Component 2
- `two_stage_metrics_tidy.csv`
- `two_stage_predictions_tidy.csv`
- `two_stage_stage2_coefficients_tidy.csv`
- `two_stage_auc_bootstrap_tidy.csv`
- `figures_component2/`

## Requirements
Python with:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy` (optional, for KDE plots)

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
