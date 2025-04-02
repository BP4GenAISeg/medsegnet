import string
from typing import Any, Dict, Type
from omegaconf import DictConfig
import torch
import torch.nn as nn

from models.base import ModelBase
from . import _MODEL_REGISTRY

def create_model(arch_cfg: DictConfig, model_type: str) -> ModelBase:
    """Factory function for creating models based on config"""
    model_cls = _MODEL_REGISTRY.get(model_type.lower())
    assert model_cls is not None, f"Model type {model_type} not registered."
    return model_cls.from_config(arch_cfg)
