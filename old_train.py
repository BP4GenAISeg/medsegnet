import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from omegaconf import DictConfig
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra

from models.unet3d import UNet3D
from data.datasets import HepaticVesselDataset
from utils.experimentmanager import ExperimentManager
from utils.metrics import dice_score
from preprocessing.dimensions import precompute_dimensions
from utils.task import extract_task_name

# Training Loop
@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    exp_manager = ExperimentManager(cfg, model_name="unet3d", task_name="Task08_hepatic_vessel")

    if cfg.gpu.mode == "multi":
        dist.init_process_group(backend=cfg.gpu.backend)
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif cfg.gpu.devices:
        device = torch.device(f"cuda:{cfg.gpu.devices[0]}")
    else:
        device = torch.device("cpu")

    model = UNet3D(
        in_channels=1,
        num_classes=cfg.training.num_classes,
        n_filters=cfg.training.n_filters,
        dropout=cfg.training.dropout,
        batch_norm=True,
    ).to(device)
    # summary(model, input_size=(1, 1, 512, 512, 48))

    if cfg.gpu.mode == "multi":
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 


    shitty_code = "datasets/Task08_HepaticVessel/imagesTr/"
    dims = precompute_dimensions(shitty_code)
    task = extract_task_name(shitty_code)
    dataset = HepaticVesselDataset(cfg, target_shape=dims[task])
    
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Initialize DiceMetric once before training starts
    # dice_metric = DiceMetric(include_background=True, reduction="mean")

    for epoch in range(cfg.training.num_epochs):
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        total_loss = 0
        total_dice = 0
        
        # dice_metric.reset()  # Reset metric at the beginning of each epoch

        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()



            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Convert predictions to probabilities and one-hot encode labels
            # labels_one_hot = F.one_hot(labels, num_classes=cfg.training.num_classes).permute(0, 4, 1, 2, 3)

            # Compute Dice Score
            # dice_metric(preds, labels_one_hot)


            total_loss += loss.item()
            # total_dice += dice

            loop.set_description(f"Epoch [{epoch+1}/{cfg.training.num_epochs}]")
            
            loop.set_postfix(
                loss=loss.item(),
                # dice=f"{dice:.4f}",
                epoch=epoch+1
            )

        avg_dice = total_dice / len(dataloader)
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")
    
    exp_manager.save(model)
    
    print(f"Model and logs saved at: {exp_manager.get_experiment_path()}")


    if cfg.gpu.mode == "multi":
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
