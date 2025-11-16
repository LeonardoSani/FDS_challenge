import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomVoter(BaseEstimator, ClassifierMixin):
    """
    Custom ensemble voting classifier that handles different feature sets for different models.
    
    This classifier trains different models on different feature representations and
    combines their predictions using weighted averaging.
    
    Parameters:
    -----------
    estimators : list of (name, estimator) tuples
        List of base estimators to use for voting
    weights : array-like, optional
        Weights for each estimator. If None, equal weights are used
    """
    
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        self.normalized_weights_ = None
        self.trained_models_ = {}

    def fit(self, X_combined_fold, y_fold):
        """
        Fit the ensemble model on the training data.
        
        Parameters:
        -----------
        X_combined_fold : dict
            Dictionary containing different feature representations:
            - 'main': Features for logistic regression with PCA
            - 'tree': Features for tree-based models
        y_fold : array-like
            Target labels
            
        Returns:
        --------
        self : CustomVoter
            Fitted ensemble model
        """
        # Calculate normalized weights
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
        """
        Predict class probabilities using weighted ensemble voting.
        
        Parameters:
        -----------
        X_combined_fold : dict
            Dictionary containing different feature representations
            
        Returns:
        --------
        array-like : Weighted average class probabilities
        """
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
        weights_reshaped = self.normalized_weights_.reshape(-1, 1, 1)
        weighted_probas = probas_array * weights_reshaped
        final_probas = np.sum(weighted_probas, axis=0)
        
        return final_probas

    def predict(self, X_combined_fold):
        """
        Predict class labels using the ensemble model.
        
        Parameters:
        -----------
        X_combined_fold : dict
            Dictionary containing different feature representations
            
        Returns:
        --------
        array-like : Predicted class labels
        """
        avg_probas = self.predict_proba(X_combined_fold)
        return np.argmax(avg_probas, axis=1)

    def get_params(self, deep=True):
        """
        Get parameters for the estimator (required for scikit-learn compatibility).
        
        Parameters:
        -----------
        deep : bool, optional
            Whether to return parameters of sub-estimators
            
        Returns:
        --------
        dict : Parameter names and their values
        """
        return {
            "estimators": self.estimators,
            "weights": self.weights
        }

    def set_params(self, **params):
        """
        Set parameters for the estimator (required for scikit-learn compatibility).
        
        Parameters:
        -----------
        **params : dict
            Parameter names and their new values
            
        Returns:
        --------
        self : CustomVoter
            Updated estimator instance
        """
        if 'estimators' in params:
            self.estimators = params['estimators']
        if 'weights' in params:
            self.weights = params['weights']
        return self
    