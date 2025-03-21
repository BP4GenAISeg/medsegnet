import time
from omegaconf import DictConfig
import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from data.datasets import MedicalDecathlonDataset
from utils.inferenceutils import get_weights_depth, weighted_softmax
from utils.metrics import dice_coefficient
from utils.utils import RunManager


class Trainer:
  def __init__(
        self, 
        arch_cfg: DictConfig,
        model: torch.nn.Module, 
        train_dataloader: DataLoader[MedicalDecathlonDataset],
        val_dataloader: DataLoader[MedicalDecathlonDataset],
        test_dataloader: DataLoader[MedicalDecathlonDataset],
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device, 
        run_manager: RunManager
      ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.rm = run_manager
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.patience = arch_cfg.training.patience
        self.num_epochs = arch_cfg.training.num_epochs
        self.num_classes = arch_cfg.dataset.num_classes
        # Define weights for deep supervision outputs
        self.ds_weights = get_weights_depth(arch_cfg.model.depth)
        print("DELETE: ", self.ds_weights)
        self.rm.info(f"Trainer initialized with {self.num_epochs} epochs and patience {self.patience}")
        
  def train_one_epoch(self, epoch: int):
    """
    Train the model for one epoch.
    """
    self.model.train()
    total_loss = 0
    progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

    try:
      for images, labels in progress_bar:
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
    
        outputs = self.model(images)  # Returns (final, ds2, ds3, ds4) ONLY during training (otherwise normal)
        loss = sum((weight * self.criterion(output, labels)
                    for weight, output in zip(self.ds_weights, outputs)), 
                        torch.tensor(0.0, device=self.device))
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.rm.warning(f"Loss is NaN or Inf at epoch {epoch}, skipping this batch.", stdout=True)
            continue
        
        loss.backward()
        self.optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    except Exception as e:
      self.rm.error(f"Error during training: {str(e)}", stdout=True)
      raise e
    return total_loss / (len(self.train_dataloader) * sum(self.ds_weights))
  
  def train(self):
    """
    Main training loop.
    """
    start_time = time.time()

    num_epochs = self.num_epochs
    for epoch in trange(num_epochs, desc="Training Progress"):
      train_loss = self.train_one_epoch(epoch)
      val_loss = self.validate()
      
      self.rm.info(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", stdout=True)

      is_improved = val_loss < self.best_val_loss
      if is_improved:
          self.best_val_loss = val_loss
          self.early_stopping_counter = 0
          self.rm.save_model(self.model)
          self.rm.info(f"New best model saved with validation loss: {val_loss:.4f}")
      else:
          self.early_stopping_counter += 1
        
      if self.early_stopping_counter >= self.patience:
          self.rm.info(f"Early stopping triggered after {self.patience} epochs without improvement.", stdout=True)
          break

    total_time = time.time() - start_time
    self.rm.info(f"Training completed in {total_time:.2f} seconds.", stdout=True)


  def validate(self):
      """Evaluates the model on the validation set."""
      self.model.eval()
      val_loss = 0.0
      dice_score_total = 0.0

      with torch.no_grad():
          try:
              for images, masks in self.val_dataloader:
                  images, masks = images.to(self.device), masks.to(self.device)

                  outputs = self.model(images)
                  
                  loss = sum(
                      (weight * self.criterion(output, masks) for weight, output in zip(self.ds_weights, outputs)),
                      torch.tensor(0.0, device=self.device)
                  ) if len(outputs) > 1 else self.criterion(outputs, masks)
                  
                  if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Invalid validation loss: {loss.item()}")

                  val_loss += loss.item()

                  dice_score = dice_coefficient(outputs, masks, self.num_classes)
                  # dice_score_total += dice_score.item() TODO´- tænker den skal slettes
                  dice_score_total += dice_score
          except Exception as e:
              self.rm.error(f"Error during validation: {str(e)}", stdout=True)
              raise e
      avg_val_loss = val_loss / len(self.val_dataloader)
      avg_dice_score = dice_score_total / len(self.val_dataloader)
      self.rm.info(f"Validation Loss: {avg_val_loss:.4f} | Dice Score: {avg_dice_score:.4f}", stdout=True)
      return avg_val_loss


  def test(self):
    """Tests the model on the test set."""
    self.rm.load_model(self.model)    
    self.model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, masks in self.test_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            test_loss += loss.item()
    avg_test_loss = test_loss / len(self.test_dataloader)
    self.rm.info(f"Test Loss: {avg_test_loss:.4f}", stdout=True)