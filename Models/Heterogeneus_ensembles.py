import pandas as pd
import numpy as np

# --- Preprocessing & Pipeline ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Base Models ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- Ensemble Methods ---
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# --- Model Selection & Evaluation ---
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomVoter(BaseEstimator, ClassifierMixin):
    
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        # --- CHANGE 1: Store the original weights ---
        # Do not modify the parameter passed in.
        self.weights = weights
        
        # We'll store the calculated, normalized weights in a
        # separate attribute with a trailing underscore.
        self.normalized_weights_ = None
        
        self.trained_models_ = {}

    def fit(self, X_combined_fold, y_fold):
        
        # --- CHANGE 2: Calculate normalized weights in fit() ---
        # This ensures normalization happens, but outside the
        # __init__ constructor, so it doesn't break cloning.
        if self.weights is None:
            self.normalized_weights_ = np.ones(len(self.estimators)) / len(self.estimators)
        else:
            self.normalized_weights_ = np.array(self.weights) / np.sum(self.weights)
        
        
        self.trained_models_ = {} 
        X_data_map = {
            'lr_pca': X_combined_fold['main'],
            'rf': X_combined_fold['tree'],
            'xgb': X_combined_fold['tree']
        }
        for name, model_pipeline in self.estimators:
            fitted_model = model_pipeline.fit(X_data_map[name], y_fold)
            self.trained_models_[name] = fitted_model
            
        return self
    
    def predict_proba(self, X_combined_fold):
        all_probas = []
        X_data_map = {
            'lr_pca': X_combined_fold['main'],
            'rf': X_combined_fold['tree'],
            'xgb': X_combined_fold['tree']
        }
        for name, model in self.trained_models_.items():
            probas = model.predict_proba(X_data_map[name])
            all_probas.append(probas)
            
        probas_array = np.array(all_probas)
        
        # --- CHANGE 3: Use the new normalized_weights_ ---
        weights_reshaped = self.normalized_weights_.reshape(-1, 1, 1)
        
        weighted_probas = probas_array * weights_reshaped
        final_probas = np.sum(weighted_probas, axis=0)
        
        return final_probas

    def predict(self, X_combined_fold):
        avg_probas = self.predict_proba(X_combined_fold)
        return np.argmax(avg_probas, axis=1)

    # --- CHANGE 4 (CRITICAL): get_params() ---
    # This is what scikit-learn uses to clone.
    # It must return the *original*, *unmodified* values
    # that were passed to __init__.
    def get_params(self, deep=True):
        return {
            "estimators": self.estimators,
            "weights": self.weights  # Return the original, unmodified 'weights'
        }

    # You also need set_params to be fully compliant
    def set_params(self, **params):
        if 'estimators' in params:
            self.estimators = params['estimators']
        if 'weights' in params:
            self.weights = params['weights']
        return self
    