from pathlib import Path
import yaml
from schemas.config_schema import (
    APIFetchConfig,
    Config,
    BugzillaConfig,
    GlobalConfig,
    GlobalDatasetStoreConfig,
    SearchConfig,
    DatasetStoreConfig,
    LogConfig,
)


def get_root_dir(root_file: str) -> Path:
    """
    Find the project root directory containing the specified file.

    Raises:
        FileNotFoundError: If no parent directory contains the file.
    """
    root = next(
        (p for p in Path(__file__).resolve().parents if (p / root_file).exists()),
        None,
    )
    if root is None:
        raise FileNotFoundError(f"Could not find project root containing {root_file}")
    return root


# Paths to project and global configuration files
PROJECT_ROOT: Path = get_root_dir("main.py")
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

GLOBAL_ROOT: Path = get_root_dir("pyproject.toml")
GLOBAL_CONFIG_PATH: Path = GLOBAL_ROOT / "global_config.yaml"


def load_config() -> Config:
    """
    Load the local project config from YAML and return a Config object.
    """
    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        bugzilla_config=BugzillaConfig(**raw["bugzilla_config"]),
        search_config=SearchConfig(**raw["search_config"]),
        api_fetch_config=APIFetchConfig(**raw["api_fetch_config"]),
        dataset_store_config=DatasetStoreConfig(**raw["dataset_store_config"]),
        log_config=LogConfig(**raw["log_config"]),
    )


def load_global_config() -> GlobalConfig:
    """
    Load the global project config from YAML and return a GlobalConfig object.
    """
    with GLOBAL_CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return GlobalConfig(
        dataset_store_config=GlobalDatasetStoreConfig(**raw["dataset_store_config"])
    )


# Load configuration once at module import
config: Config = load_config()
global_config: GlobalConfig = load_global_config()
