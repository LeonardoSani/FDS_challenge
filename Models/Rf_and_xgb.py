from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_model_pipeline_rf(random_state=0, **model_kwargs):
    """
    Creates a RandomForest pipeline that accepts model hyperparameters directly.

    Args:
        random_state (int): Random seed for reproducibility.
        **model_kwargs: Additional keyword arguments for RandomForestClassifier.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **model_kwargs
    )

    pipeline = Pipeline([
        ('model', model)
    ])
    return pipeline


def create_model_pipeline_xgb(random_state=0, **model_kwargs):
    """
    Creates an XGBoost pipeline that accepts model hyperparameters directly.

    Args:
        random_state (int): Random seed for reproducibility.
        **model_kwargs: Additional keyword arguments for XGBClassifier.

    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    model = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',  # avoids XGBoost warning about deprecated defaults
        **model_kwargs
    )

    pipeline = Pipeline([
        ('model', model)
    ])
    return pipeline

def get_feature_importance(model_pipeline, feature_names, top_k, plot=False):
    """
    Extracts and returns feature importances from a trained pipeline.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): A trained pipeline containing a model
                                                     with a 'feature_importances_' attribute 
                                                     (e.g., RandomForest or XGBoost).
        feature_names (list or pd.Index): The names of the features used for training.
        top_k (int): The number of top features to return and plot.
        plot (bool): Whether to display a plot of feature importances. Defaults to False.

    Returns:
        pd.Series: A series of top_k feature importances, sorted in descending order.
    """
    # Access the trained model step from the pipeline
    model = model_pipeline.named_steps['model']

    # Get feature importances from the trained model
    importances = model.feature_importances_

    # Create a pandas Series for better visualization
    feature_importance_series = pd.Series(importances, index=feature_names)

    # Sort features by importance and get top_k
    sorted_importances = feature_importance_series.sort_values(ascending=False)
    top_features = sorted_importances.head(top_k)
    
    # Handle plotting if requested
    if plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features.values, y=top_features.index, orient='h')
        plt.title(f'Top {top_k} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Ensure the plot is displayed in Jupyter notebooks
        try:
            from IPython.display import display
            plt.show()
        except ImportError:
            plt.show()
    
    return top_features