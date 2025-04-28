from sys import stderr
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from data.data_manager import DataManager

# from trainer import Trainer
from trainer_ms import Trainer
from data.datasets import (
    MedicalDecathlonDataset,
    VALID_TASKS,
    ProstateDataset,
    BrainTumourDataset,
)
from utils.assertions import ensure_has_attr, ensure_has_attrs
from utils.losses import get_loss_fn
from utils.utils import RunManager, prepare_dataset_config, setup_logging, setup_seed
import random
import numpy as np
from models.factory import create_model
from utils.wandb_logger import get_wandb_logger
import argparse
import logging
from hydra.utils import instantiate


EXCLUDED_TASKS = {"Task01_BrainTumour", "Task05_Prostate"}
DATASET_MAPPING = {
    task: MedicalDecathlonDataset for task in VALID_TASKS - EXCLUDED_TASKS
}
DATASET_MAPPING["Task01_BrainTumour"] = BrainTumourDataset
DATASET_MAPPING["Task05_Prostate"] = ProstateDataset


# Husk hydra.utils.instantiate MED HYDRAS _target_ CONVENTION!


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ensure_has_attrs(cfg, ["active_dataset", "active_architecture", "gpu"], Exception)

    seed = cfg.seed
    setup_seed(seed)

    task_name = cfg.active_dataset
    model_name = cfg.active_architecture

    assert task_name in DATASET_MAPPING, f"Unknown dataset: {task_name}"

    run_manager = RunManager(cfg)
    setup_logging(cfg.get("logging", {}), run_manager.run_exp_dir)

    logger = logging.getLogger(__name__)

    gpu_device = cfg.gpu.devices[0]  # TODO: Handle multiple GPUs
    device = (
        torch.device(f"cuda:{gpu_device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = create_model(cfg, model_name).to(device)

    try:
        # TODO: (Performace boost, but doesn't change anything)try the  =>  params=filter(lambda p: p.requires_grad, model.parameters())
        # url: https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598
        # but you would also need to reinitialize the optimizer, in case of new unfrozen layers
        # and the optimizer would need to be reinitialized, so adds more complexity
        optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
        criterion = instantiate(cfg.training.loss)
        lr_scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer)
    except Exception as e:
        logger.error(f"Failed to instantiate network: {e}")
        exit(1)

    dataset_class = DATASET_MAPPING[task_name]
    data_manager = DataManager(
        dataset_class, cfg, seed, split_ratios=(0.80, 0.05, 0.15)
    )
    train_dataloader, val_dataloader, test_dataloader = data_manager.get_dataloaders()

    wandb_logger = get_wandb_logger(cfg=cfg, model=model)

    trainer = Trainer(
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        run_manager,
        wandb_logger,
    )

    final_status_code = 0
    try:
        trainer.train()
        trainer.test()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        final_status_code = 1
    finally:
        if wandb_logger:
            wandb_logger.finalize(exit_code=final_status_code)
        logger.info("Training completed.")


if __name__ == "__main__":
    main()
