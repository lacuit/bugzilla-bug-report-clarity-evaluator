from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from config.config import config

RANDOM_STATE = config.model_features_config.random_state


def get_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        solver="saga",
        class_weight="balanced",
        C=1.0,
        l1_ratio=0,
        random_state=RANDOM_STATE,
    )


def get_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def get_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        batch_size=64,
        random_state=RANDOM_STATE,
    )


def get_catboost(scale_pos_weight: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        eval_metric="Logloss",
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE,
        task_type="CPU",
        verbose=0,
        early_stopping_rounds=50,
    )


def get_lightgbm(scale_pos_weight: float) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
