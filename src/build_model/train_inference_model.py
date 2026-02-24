import logging
import os

import numpy as np
from config.config import GLOBAL_ROOT, config
from data.load_data import load_dataset
from models.train_cv_models import cross_validate_models
from preprocessing.preprocess import prepare_features
from models.models import get_lightgbm

logging.basicConfig(
    level=config.log_config.level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    SAVE_MODEL_DIR = GLOBAL_ROOT / config.model_save_config.store_dir
    SAVE_MODEL_PATH = SAVE_MODEL_DIR / "model.txt"
    SAVE_MODEL_META_PATH = SAVE_MODEL_DIR / "model_meta.txt"

    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    if SAVE_MODEL_PATH.exists():
        logger.info(
            f"Please move/remove model save before runnning again at {SAVE_MODEL_PATH}"
        )
        raise FileExistsError("Model will be overwritten")

    logger.info("Training Final Model")

    df = load_dataset()

    # Feature preparation
    X, y, scale_pos_weight = prepare_features(df)

    model_label_cv = "final_model_cv"
    models = {model_label_cv: get_lightgbm(scale_pos_weight)}

    # Run 5-fold CV to find optimal threshold
    results = cross_validate_models(
        models=models,
        X=X,
        y=y,
        n_splits=5,
    )

    metrics = results[model_label_cv]
    logger.info(
        f"Mean ROC-AUC: {metrics['mean_roc_auc']:.3f} | "
        f"Mean F1: {metrics['mean_f1']:.3f} | "
        f"Mean Thresh: {metrics['mean_threshold']:.2f}"
    )

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    final_scale_pos_weight = n_neg / n_pos

    final_model = get_lightgbm(final_scale_pos_weight)
    final_model.fit(X, y)
    final_model.booster_.save_model(SAVE_MODEL_PATH)

    with open(SAVE_MODEL_META_PATH, "w") as f:
        f.write(str(metrics["mean_threshold"]))

    logger.info("Model Saved")
    logger.info(f"Threshold for model is {metrics['mean_threshold']}")


if __name__ == "__main__":
    main()
