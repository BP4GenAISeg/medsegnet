from omegaconf import DictConfig
from torch.optim import NAdam
from optimizers import register_optimizer
from pydantic import BaseModel

class NAdamParams(BaseModel):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

@register_optimizer("nadam")
class NAdamOptimizer:
    def __init__(self, params, config: NAdamParams): 
        self.optimizer = NAdam(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    def __call__(self):
        return self.optimizer
    
