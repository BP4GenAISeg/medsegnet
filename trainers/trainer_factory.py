from .base_trainer import BaseTrainer
from .deep_supervision_trainer import DeepSupervisionTrainer
from .multiscale_trainer import MultiscaleTrainer


def get_trainer(cfg, *args, **kwargs):
    arch = cfg.architecture.name
    if arch == "ds-unet3d":
        return DeepSupervisionTrainer(cfg, *args, **kwargs)
    elif arch == "ms-unet3d":
        return MultiscaleTrainer(cfg, *args, **kwargs)
    else:
        return BaseTrainer(cfg, *args, **kwargs)
