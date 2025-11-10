from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

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
# add feature importance