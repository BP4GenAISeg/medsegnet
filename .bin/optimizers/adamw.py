from networkx import prominent_group
from omegaconf import DictConfig
from torch.optim import AdamW
from optimizers import register_optimizer
from pydantic import BaseModel

class AdamWParams(BaseModel):
    lr: float = 3e-4  
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-4

@register_optimizer("adamw")
class AdamWOptimizer:
    def __init__(self, params, config: AdamWParams): 
        self.optimizer = AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        print(config.weight_decay)
    def __call__(self):
        return self.optimizer
    
