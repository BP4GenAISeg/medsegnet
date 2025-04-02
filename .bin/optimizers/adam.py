from omegaconf import DictConfig
from torch.optim import Adam
from optimizers import register_optimizer
from pydantic import BaseModel

class AdamParams(BaseModel):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

@register_optimizer("adam")
class AdamOptimizer:
    def __init__(self, params, config: AdamParams): 
        self.optimizer = Adam(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )

    def __call__(self):
        return self.optimizer
    
