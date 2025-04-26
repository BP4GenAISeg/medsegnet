# utils/callbacks.py (or wherever you prefer)
import numpy as np
import torch

from utils.assertions import ensure_in
from utils.utils import RunManager
import logging

class EarlyStopping:
    """
    Monitors a validation metric and stops training when it stops improving.
    Tracks best scores and determines improvement based on configured criterion.
    """
    def __init__(self, patience:int, delta:float, criterion:str, verbose:bool):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            criterion (str): Metric to monitor ('loss', 'dice', 'both').
            verbose (bool): If True, prints messages about improvement/stopping.
            run_manager (RunManager): RunManager instance for logging messages.
        """
        ensure_in(criterion, ['loss', 'dice', 'both'], KeyError)

        self.patience = patience
        self.delta = delta
        self.criterion = criterion
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False

        self.best_loss = np.Inf
        self.best_dice = -np.Inf
        self.logger = logging.getLogger(__name__)


    def __call__(self, val_loss, val_dice):
        """
        Checks if metrics improved and updates counter/stop flag.
        """
        previous_best_loss, previous_best_dice = self.best_loss, self.best_dice

        improvement_checks = {
            'loss': lambda: val_loss < self.best_loss - self.delta,
            'dice': lambda: val_dice > self.best_dice + self.delta,
            'both': lambda: (val_loss < self.best_loss - self.delta) \
                        and (val_dice > self.best_dice + self.delta)
        }

        # --- Update best scores if improved ---
        improved = improvement_checks[self.criterion]()
        if improved:
            self.best_loss, self.best_dice = val_loss, val_dice
            self.counter = 0
            
            if self.verbose:
                messages = {
                  'loss': f"Validation loss improved ({previous_best_loss:.4f} -> {val_loss:.4f})",
                  'dice': f"Validation dice improved ({previous_best_dice:.4f} -> {val_dice:.4f})",
                  'both': f"Both improved (loss: {previous_best_loss:.4f}->{val_loss:.4f}, dice: {previous_best_dice:.4f}->{val_dice:.4f})"
                }
                self.logger.info(messages[self.criterion])
        else:
            self.counter += 1
            if self.verbose:
                crit_name = "loss & dice" if self.criterion == "both" else self.criterion
                self.logger.info(f'No improvement in {crit_name} ({self.counter}/{self.patience})')
                
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f'Early stopping triggered')

        return self.early_stop, improved