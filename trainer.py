import time
from omegaconf import DictConfig
import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

class Trainer:
  def __init__(
        self, 
        training_cfg: DictConfig, 
        model: torch.nn.Module, 
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: torch.utils.data.Dataset,
        device: torch.device, 
      ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.patience = training_cfg.patience
        self.num_epochs = training_cfg.num_epochs

  def train_one_epoch(self, epoch):
    """
    Train the model for one epoch.
    """
    self.model.train()
    total_loss = 0
    progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

    for images, labels in progress_bar:
      images, labels = images.to(self.device), labels.to(self.device)
      self.optimizer.zero_grad()

      outputs = self.model(images)
      loss = self.criterion(outputs, labels)
      
      loss.backward()
      self.optimizer.step()
      
      total_loss += loss.item()
      progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(self.train_dataloader)
  
  def train(self):
    """
    Main training loop.
    """
    start_time = time.time()
    num_epochs = self.num_epochs
    for epoch in trange(num_epochs, desc="Training Progress"):
      train_loss = self.train_one_epoch(epoch)
      val_loss = self.validate()


      print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

      is_improved = val_loss < self.best_val_loss
      if is_improved:
          self.best_val_loss = val_loss
          self.early_stopping_counter = 0
          #TODO: Save model (current best)
      else:
          self.early_stopping_counter += 1
        
      if self.early_stopping_counter >= self.patience:
          print(f"Early stopping triggered after {self.patience} epochs without improvement.")
          break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")


  def validate(self):
      """Evaluates the model on the validation set."""
      self.model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for images, masks in self.val_dataloader:
              images, masks = images.to(self.device), masks.to(self.device)

              outputs = self.model(images)
              loss = self.criterion(outputs, masks)
              
              val_loss += loss.item()
      return val_loss / len(self.val_dataloader)
  

  def test(self):
    """Tests the model on the test set."""
    #TODO: Load best model
    self.model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for images, masks in self.test_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            test_loss += loss.item()
    avg_test_loss = test_loss / len(self.test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")