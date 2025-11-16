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


## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/FDS_Challenge.git
cd FDS_Challenge

```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
If you use uv:
```bash
uv sync
```

### 3. Run the Notebook

Open Notebook.ipynb to explore:
-	EDA
-	Feature creation
-	Model training
-	Performance comparison
-	Visualizations

### 4. Using the Feature Extraction Module

**Example (pseudo-code):**

```python
from extractors import get_features
from Aggregator import FeatureAggregator

# Extract raw features from the battle data
features = get_features(data)

# Initialize the feature aggregator
agg = FeatureAggregator(features)

# Build the feature set for a specific model (e.g., XGBoost)
X = agg.build_feature_set("xgb")
```
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

##  Authors

- **Simone Mantero**  
- **Leonardo Sani**  
- **Kian Sorooshmehr**


