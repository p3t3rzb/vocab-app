from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    hidden_size: int = 16
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    val_split: float = 0.2
    seed: int = 42


@dataclass
class PredictConfig:
    recall_threshold: float = 0.8
    bisect_steps: int = 16
    initial_delta_seconds: float = 86_400.0  # 1 day starting upper-bound guess
    max_delta_seconds: float = 31_536_000.0  # 1-year cap
    poly_degree: int = 2
