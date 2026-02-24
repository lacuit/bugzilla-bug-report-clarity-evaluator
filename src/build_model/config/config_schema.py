from dataclasses import dataclass


### Build Model Config
@dataclass
class ModelFeaturesConfig:
    textual: list[str]
    categorical: list[str]
    random_state: int


@dataclass
class ModelSaveConfig:
    store_dir: str


@dataclass
class EmbeddingsConfig:
    model_to_load: str
    content_chunk_size: float
    word_to_token_ratio: float
    encode_batch_size: int


@dataclass
class DatasetConfig:
    dataset_sample: float


@dataclass
class LogConfig:
    level: str


@dataclass
class Config:
    model_features_config: ModelFeaturesConfig
    embeddings_config: EmbeddingsConfig
    dataset_config: DatasetConfig
    model_save_config: ModelSaveConfig
    log_config: LogConfig


### Global Config
@dataclass
class GlobalDatasetStoreConfig:
    output_path_parquet: str


@dataclass
class GlobalConfig:
    dataset_store_config: GlobalDatasetStoreConfig
