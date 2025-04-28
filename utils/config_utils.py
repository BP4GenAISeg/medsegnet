from omegaconf import DictConfig

def get_common_args(config: DictConfig) -> dict:
    """
    Extracts common arguments from the configuration for model initialization. For UNet 3d and dynamic UNet 3d.
    """
    return {
        'in_channels': config.model.in_channels,
        'num_classes': config.dataset.num_classes,
        'n_filters': config.model.n_filters,
        'dropout': config.training.dropout,
        'batch_norm': config.training.batch_norm,
        'ds': config.model.deep_supervision.enabled,
        'ms': config.model.deep_supervision.multi_scaling,
        'inference_fusion_mode': config.model.deep_supervision.inference_fusion_mode
    }