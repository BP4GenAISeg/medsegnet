from tqdm import trange, tqdm

class Trainer:
  def __init__(self, cfg, model, dataloader, criterion, optimizer, dataset, device):
    self.cfg = cfg
    self.model = model
    self.dataloader = dataloader
    self.criterion = criterion
    self.optimizer = optimizer
    self.dataset = dataset
    self.device = device

  def train_one_epoch(self, epoch):
    self.model.train()
    total_loss = 0
    progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.cfg.training.num_epochs}", leave=False)

    for images, labels in progress_bar:
      images, labels = images.to(self.device), labels.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(images)
      loss = self.criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()
      total_loss += loss.item()

      progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(self.dataloader)
  
  def train(self):
    for epoch in trange(self.cfg.training.num_epochs, desc="Training Progress"):
      avg_loss = self.train_one_epoch(epoch)
      print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
    print("Training complete.")




    