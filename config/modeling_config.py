"""
Configuration for insurance predictive modeling
"""
MODEL_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'cv_folds': 5,
    
    'xgboost_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    
    'random_forest_params': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42
    },
    
    'feature_threshold': 0.01  # Minimum importance for feature selection
}

RISK_SCORES = {
    'province': {
        'Gauteng': 1.4,
        'KwaZulu-Natal': 1.2,
        'Limpopo': 1.1,
        'North West': 0.9,
        'Mpumalanga': 0.9,
        'Western Cape': 0.8
    },
    
    'cover_type': {
        'Comprehensive': 1.0,
        'Third Party Fire & Theft': 0.8,
        'Third Party': 0.7
    }
}