# models/__init__.py
from typing import Dict, Type
import importlib
import pkgutil
from typing import Dict, Type
from .base import ModelBase
import pkgutil
import importlib
        
_MODEL_REGISTRY: Dict[str, Type[ModelBase]] = {}

def register_model(name: str):
    """Class decorator to register models in the factory registry."""
    def decorator(cls: Type[ModelBase]):
        assert issubclass(cls, ModelBase), "Only ModelBase subclasses can be registered"
        assert hasattr(cls, 'from_config'), f"Model {name} must implement from_config method"
        assert name.lower() not in _MODEL_REGISTRY, f"Model name {name} already registered"

        _MODEL_REGISTRY[name.lower()] = cls
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
