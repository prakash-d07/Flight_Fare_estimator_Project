from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DatapreprocessConfig:
    root_dir: Path
    datapath: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    datapath: Path

@dataclass(frozen=True)
class DataModellingConfig:
    root_dir: Path
    x_datapath: Path
    y_datapath: Path
    model_file_path: Path
    preprocessor_file_path: Path
    max_depth: float
    max_features: str
    min_samples_leaf: float
    min_samples_split: float
    n_estimators: float

