from abc import ABC, abstractmethod
from typing import Sequence
import torch

class BaseFusion(ABC):
    """
    Abstract base class for fusion strategies.
    """

    def __init__(self, weights:list[float]):
        self.weights = weights

    @abstractmethod
    def fuse(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Abstract method to fuse outputs.

        Args:
            outputs (Sequence[torch.Tensor]): A sequence of output tensors.

        Returns:
            torch.Tensor: The fused output tensor.
        """
        raise NotImplementedError("Subclasses must implement this method.")