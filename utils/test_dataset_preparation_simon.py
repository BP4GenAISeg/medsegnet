from omegaconf import DictConfig, OmegaConf
# det her er lige så vi kan teste, hvsi or når vi laver store ændringer men den fkn config XD
# (nok primært mig ups)


def prepare_dataset_config(cfg: DictConfig) -> DictConfig:
    """
    Prepares a unified configuration by merging architecture-specific defaults,
    training defaults, and dataset-specific settings for the active dataset.
    """
    if "active_dataset" not in cfg or cfg.active_dataset not in cfg.datasets:
        raise KeyError(f"Active dataset '{cfg.active_dataset}' not found in datasets.")
    
    if "active_architecture" not in cfg or cfg.active_architecture not in cfg.architectures:
        raise KeyError(f"Active architecture '{cfg.active_architecture}' not found in architectures.")
    
    dataset_cfg: DictConfig = cfg.datasets[cfg.active_dataset]
    arch_cfg: DictConfig = cfg.architectures[cfg.active_architecture]

    # Merge model defaults with any dataset-specific model overrides
    # model_overrides = dataset_cfg.get(f"{cfg.active_architecture}_overrides", {}).get("model", {})
    merged_model = OmegaConf.merge(arch_cfg.model_defaults, dataset_cfg.overrides.model)

    merged_training = OmegaConf.merge(arch_cfg.training_defaults, dataset_cfg.overrides.training)
    # training_overrides = dataset_cfg.get(f"overrides", {}).get("training", {})
    # merged_training = OmegaConf.merge(arch_cfg.get("training_defaults", {}), training_overrides)

    unified_cfg = OmegaConf.create({
        "model": merged_model,
        "training": merged_training,
        "dataset": OmegaConf.create({
            # Clean the dataset config by removing architecture-specific overrides
            k: v for k, v in dataset_cfg.items()
            if k != f"overrides"
        })
    })

    print(unified_cfg)

    if not isinstance(unified_cfg, DictConfig):
        raise TypeError("Unified config is not a DictConfig. Check your configuration structure.")

    return unified_cfg

# Example usage
if __name__ == "__main__":
    import yaml
    with open("conf/config.yaml", "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    try:
        unified_cfg = prepare_dataset_config(cfg)
        print(OmegaConf.to_yaml(unified_cfg))
    except KeyError as e:
        print(f"Configuration error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")