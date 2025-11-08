from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def create_model_pipeline_rf(n_estimators, max_depth, min_samples_split, max_features, random_state=0):
    """
    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    
    # Create a list of (name, transformer) tuples
    steps = [
        ('model', RandomForestClassifier(n_estimators=n_estimators, 
                                      max_depth=max_depth,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=random_state, 
                                      n_jobs=-1))
    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline




def create_model_pipeline_xgb(random_state):
    """
    Returns:
        sklearn.pipeline.Pipeline: The unfitted model pipeline.
    """
    
    # Create a list of (name, transformer) tuples
    steps = [

    ('model', XGBClassifier(random_state=42, n_jobs=-1))

    ]
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


# add feature importance