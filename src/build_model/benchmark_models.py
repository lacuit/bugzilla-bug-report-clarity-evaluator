import logging
import polars as pl

from config.config import config
from data.load_data import load_dataset
from models.train_cv_models import get_models, cross_validate_models
from preprocessing.preprocess import prepare_features
from preprocessing.param import TARGET_COL

logging.basicConfig(
    level=config.log_config.level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting benchmarking pipeline")

    df = load_dataset().sample(fraction=config.dataset_config.dataset_sample)

    logger.info("Target distribution:")
    logger.info(df.group_by(TARGET_COL).agg(pl.count()))

    # Feature preparation
    X, y, scale_pos_weight = prepare_features(df)

    # Get models
    models = get_models(scale_pos_weight)

    # Run 5-fold CV
    results = cross_validate_models(
        models=models,
        X=X,
        y=y,
        n_splits=5,
    )

    logger.info("\n===== MODEL COMPARISON (5-Fold CV) =====")
    for name, metrics in sorted(
        results.items(), key=lambda x: x[1]["mean_f1"], reverse=True
    ):
        logger.info(
            f"{name:20s} | "
            f"Mean ROC-AUC: {metrics['mean_roc_auc']:.3f} | "
            f"Mean F1: {metrics['mean_f1']:.3f} | "
            f"Mean Thresh: {metrics['mean_threshold']:.2f}"
        )

    logger.info("Benchmarking complete.")


if __name__ == "__main__":
    main()
