# models/__init__.py
from pyexpat import model
from typing import Callable, Dict, Type
import importlib
import pkgutil
from typing import Dict, Type, Any, cast
import pkgutil
import importlib
from models.base import ModelBase
from pydantic import BaseModel, ValidationError
from omegaconf import DictConfig, OmegaConf
import inspect
from utils.assertions import ensure, ensure_has_attr, ensure_has_attrs, ensure_in, ensure_is_instance
from torch.optim import Optimizer

_OPTIMIZERS_REGISTRY: dict[str, tuple[type, type[BaseModel]]] = {}

def register_optimizer(name: str):
    def decorator(cls):
        init_signature = inspect.signature(cls.__init__)
        param_type = init_signature.parameters['config'].annotation
        _OPTIMIZERS_REGISTRY[name.lower()] = (cls, param_type)
        return cls
    return decorator

for _, module_name, _ in pkgutil.walk_packages(
    path=__path__,  
    prefix=f"{__name__}.",  
    onerror=lambda _: None  
):
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import {module_name}: {str(e)}")


def get_optimizer(model: ModelBase, train_cfg: DictConfig) -> Optimizer:
    """
    Factory function to create an optimizer with full type safety and validation
    """
    try:
        ensure_has_attr(train_cfg, "optimizer", KeyError)
        optim_cfg = train_cfg.optimizer
        
        ensure_has_attr(optim_cfg, "name", KeyError)
        lower_name = str(optim_cfg.name).lower()

        ensure_in(lower_name, _OPTIMIZERS_REGISTRY, KeyError)
        optim_cls, param_model = _OPTIMIZERS_REGISTRY[lower_name]

        # Convert OmegaConf to Python native types
        # params_dict = OmegaConf.to_container(optim_cfg.get("params", {}), resolve=True)
        params = optim_cfg.get("params", {})
        params_dict = OmegaConf.to_container(params, resolve=True) if OmegaConf.is_config(params) else params

        # Ensure we have a proper dictionary with string keys
        ensure_is_instance(params_dict, dict, KeyError)
            
        # Explicit type casting for type checkers
        params_dict = cast(Dict[str, Any], params_dict)
        
        # Validate parameters
        validated_params = param_model(**params_dict)
        
        optimizer_wrapper = optim_cls(model.parameters(), validated_params)

        return optimizer_wrapper()

    except ValidationError as e:
        error_msg = "\n".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        raise ValueError(f"Optimizer config validation failed:\n{error_msg}") from e