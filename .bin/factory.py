import string
from typing import Any, Dict, Type
from omegaconf import DictConfig
import torch
import torch.nn as nn

from models.base import ModelBase
from . import _MODEL_REGISTRY


def create_model(cfg: DictConfig, model_type: str) -> ModelBase:
    """Factory function for creating models based on config"""
    model_cls = _MODEL_REGISTRY.get(model_type.lower())
    if model_cls is None:
        raise ValueError(
            f"Model type '{model_type}' not found in registry. Available types: {list(_MODEL_REGISTRY.keys())}"
        )

    return model_cls(cfg)
