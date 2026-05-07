# Evaluation of Machine Learning Methods

Course exercises for **TKO_7092 – Evaluation of Machine Learning Methods (2025)**.  
Three exercises progressively teach how to design correct cross-validation strategies to apply for different cases.

---

## Exercises

### Exercise 1 – KNN Regression with Replica-Aware Cross-Validation
**Notebook:** `EMLM_exercise1_template.ipynb`  
**Data:** `water_data.xls` (225 water samples, 3 sensor inputs → 3 metal ion targets: total metal, Cd, Pb)

Predicts metal ion concentrations from electrochemical sensor measurements using K-Nearest Neighbor regression (k = 1, 3, 5, 7).

Compares two CV strategies:
- **LOOCV** (Leave-One-Out): leaves out one measurement at a time → optimistic C-index ~0.88–0.92
- **LROCV** (Leave-Replicas-Out): leaves out entire replica groups → realistic C-index ~0.71–0.82

**Key insight:** When data contains replicated measurements of the same sample, LOOCV leaks information across folds. LROCV gives a more honest estimate of generalization to new mixtures.

---

### Exercise 2 – Spatial Cross-Validation for Water Permeability
**Notebook:** `EMLM_exercise2_template.ipynb`  
**Data:** `input.xls`, `output.xls`, `coordinates.xls` (1690 samples, 75 features + 2D spatial coordinates)

Predicts water permeability using 7-NN regression and evaluates how performance degrades with geographic distance.

Implements **Spatial Leave-One-Out CV (SLOCV)**: for each test point, training points within distance threshold `d` are excluded. Threshold is varied from 0 to 300 meters.

**Key insight:** C-index drops from ~0.85 (at 0 m) to ~0.68 (at 300 m). The model is reliable up to ~100 m. Beyond that, spatial autocorrelation no longer helps and performance degrades significantly.

---

### Exercise 3 – Correcting Data Leakage in Drug-Protein Pair Prediction
**Notebook:** `EMLM_exercise3_template.ipynb`  
**Data:** `input.data`, `output.data`, `pairs.data` (400 drug-protein pairs, 77 proteins, 59 drugs)

A bio-scientist's 10-NN model achieved >90% C-index with standard LOOCV but failed in the lab on new molecules. This exercise diagnoses and fixes the problem.

**Root cause:** Standard LOOCV removes one pair but both molecules appear in training — the model has already "seen" them. Four out-of-sample scenarios are defined (A–D), and correct CV is implemented for each:
- **Leave-One-Drug-Out C-index: 0.52** → near random, drug features are memorized
- **Leave-One-Protein-Out C-index: 0.83** → model generalizes well to new proteins

**Key insight:** Standard LOOCV on pair data is almost always wrong. Group-based CV is required to simulate truly unseen molecules.

---

## Datasets

| File | Rows | Features | Used in |
|------|------|----------|---------|
| `water_data.xls` | 225 | 3 inputs, 3 targets | Exercise 1 |
| `input.xls` | 1690 | 75 | Exercise 2 |
| `output.xls` | 1690 | 1 (permeability) | Exercise 2 |
| `coordinates.xls` | 1690 | 2 (X, Y in meters) | Exercise 2 |
| `input.data` | 400 | 64 | Exercise 3 |
| `output.data` | 400 | 1 (affinity) | Exercise 3 |
| `pairs.data` | 400 | 2 (drug ID, protein ID) | Exercise 3 |

---

## Dependencies

- Python 3.x
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib openpyxl
```

---

## Core Concept

> **Proper cross-validation design is not optional.**  
> Each domain presented here, requires a CV strategy that reflects the real prediction scenario. Standard LOOCV can inflate performance estimates dramatically when data has internal structure.