from omegaconf import DictConfig, OmegaConf
from typing import Dict, Tuple


def merge_dataset_config(cfg: DictConfig) -> Tuple[DictConfig, DictConfig]:
    """
    Merges default training settings with dataset-specific overrides.
    """
    dataset_cfg: DictConfig = cfg.datasets[cfg.active_dataset]
    overrides = dataset_cfg.get("training_overrides", {})
    merged_training = OmegaConf.merge(cfg.training_defaults, overrides)
    
    if not isinstance(merged_training, DictConfig):
        raise TypeError("Merged training config is not a DictConfig. Please check your configuration structure.")
    
    return dataset_cfg, merged_training