from abc import ABC, abstractmethod
from omegaconf import DictConfig
import torch.nn as nn
from abc import ABC, abstractmethod

class ModelBase(nn.Module, ABC):  
    """
    Base class for all models in the framework.
    """
    @classmethod
    @abstractmethod  
    def from_config(cls, config: DictConfig) -> 'ModelBase':
        pass
    