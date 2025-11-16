# FDS Challenge – Pokémon Battle Winner Prediction

A Machine Learning project designed to predict the winner of a competitive simulated Pokémon battle.  
The repository contains the full workflow: feature extraction, advanced feature engineering, model training, evaluation, and the final project report.

---

##  Project Objective

The goal is to determine which player (P1 or P2) will win a Pokémon battle within the first 30 turns by analyzing:

- Pokémon identity and typing  
- Base stats and stat boosts  
- Temporal dynamics of the battle  
- Voluntary switches, faints, and damage patterns  
- Engineered game-flow features that capture competitive behavior  
##  Repository Structure

```
/
├── feature_engineering/
│   ├── __init__.py
│   ├── extractors.py
│   ├── utils.py
│   ├── constants.py
│   └── Aggregator.py
│
├── Models/
│   ├── __init__.py

│   ├── Heterogeneus_ensembles.py
│   ├── logistic_regression.py
│   ├── Rf_and_xgb.py
│   └── utils.py
│
├── Notebook.ipynb
├── FDS_Challenge_Report.pdf
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```
##  Feature Engineering

The feature engineering pipeline is designed to capture the full dynamics of a competitive Pokémon battle.  
Here are the most relevant feature families:

###  Key Features

- **Average Final HP**  
  Mean remaining HP percentage of all team Pokémon at the last observed state.

- **Voluntary Swap Count**  
  Number of voluntary switches (a switch performed while the Pokémon still has HP > 0).  
  This acts as a proxy for player style and strategic pressure.

- **Pokémon Encoding**  
  Pokémon identity encoded using one-hot encoding for linear models and label encoding for tree-based models.

---

### Feature Variants

- Temporal segmentation into **early / mid / late game**  
  - Early: turns 0–10  
  - Mid: turns 10–20  
  - Late: turns 20–30  

- **Player-to-player feature differences (P1 – P2)**  
  These reduce dimensionality while preserving strategic comparisons.

---

###  Feature Selection & Decorrelation

- **PCA** (Principal Component Analysis) for Logistic Regression.  
- **Regularization or feature importance** (Random Forest / XGBoost) for tree-based models to handle redundancy.

##  Models Used

All models were optimized using **GridSearchCV** with 10-fold cross-validation.

| Model                                 | Accuracy (CV)        |
|---------------------------------------|-----------------------|
| Logistic Regression (L2)              | 0.8504 ± 0.0065       |
| Logistic Regression (PCA + L2)        | 0.8516 ± 0.0094       |
| Logistic Regression (Polynomial)      | 0.805  ± 0.0097       |
| Random Forest                         | 0.8376 ± 0.0118       |
| **XGBoost**                           | **0.8532 ± 0.0134**   |
| Soft Voting (XGB + RF + LR-PCA)       | 0.8517 ± 0.011        |

Models selected for the private leaderboard are highlighted in the final report.





##  Results

The best-performing model is **XGBoost**, achieving an accuracy of approximately **85%**.  
The most influential features include:

- move effectiveness  
- final HP metrics  
- faint counts  
- voluntary switches  
- damage difference features  

---

##  Report

The extended report (methodology, visualizations, and detailed analysis) is available here:  
**FDS_Challenge_Report.pdf**

---

##  Project Modules Overview

The repository is organized into modular Python components that separate
feature extraction, aggregation, modeling, and utilities.  
Each module plays a specific role in the full pipeline:

- **extractors.py** – Functions to parse the raw battle data and compute
  first-level features (HP, effectiveness, switches, states per turn, etc.).

- **Aggregator.py** – Combines and activates feature groups, producing
  tailored feature sets for different model families (linear, tree-based, ensembles).

- **utils.py** – Core helper functions and domain logic  
  (type charts, base stats, dictionaries, damage utility helpers, validations).

- **Models folder** – Contains implementations for:
  - Logistic Regression (standard, PCA, polynomial)
  - Random Forest
  - XGBoost
  - Heterogeneous soft-voting ensemble  
  Each model script includes hyperparameter tuning (GridSearchCV) and evaluation tools.

---

## Notebook Description

The `Notebook.ipynb` integrates all modules into a complete workflow.  
Inside the notebook you will find:

- **Dataset exploration**  
  Visual inspection and understanding of the original battle data structure.

- **Feature explanation**  
  Practical demonstration of how each engineered feature is generated using the functions defined in the modules.

- **Model training**  
  Training routines for Logistic Regression, Random Forest, XGBoost, and the Ensemble model.

- **Validation & performance analysis**  
  Cross-validation, comparison of results, plots, and final model selection.

This notebook acts as the main interactive environment to understand,
run, and evaluate the entire project pipeline.

---

##  Authors

- **Simone Mantero**  
- **Leonardo Sani**  
- **Kian Sorooshmehr**


