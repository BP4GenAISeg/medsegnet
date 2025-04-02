# wandb_logger.py
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import os # Added for path handling


def _prepare_slice_for_wandb(tensor_3d: torch.Tensor, is_mask: bool = False, slice_dim: int = 0, slice_index: int | None = None):
    """
    Converts a 3D tensor slice (e.g., from D, H, W) to a NumPy array (H, W)
    suitable for wandb.Image. Selects the middle slice by default.
    """
    if tensor_3d.ndim != 3:
        raise ValueError(f"Expected 3D tensor (e.g., D, H, W), but got {tensor_3d.ndim}D")

    # Select slice index
    if slice_index is None:
        # Default to middle slice
        slice_index = tensor_3d.shape[slice_dim] // 2

    # Index along the specified dimension
    slices = [slice(None)] * 3
    slices[slice_dim] = slice_index
    img_slice_2d = tensor_3d[tuple(slices)] # Shape (H, W) if slice_dim=0 (Depth)

    img_np = img_slice_2d.numpy() # Already detached by indexing

    # Normalize image slice if not a mask
    if not is_mask:
        min_val, max_val = img_np.min(), img_np.max()
        if max_val > min_val:
            img_np = ((img_np - min_val) / (max_val - min_val)) * 255.0
        img_np = img_np.astype(np.uint8)
    else:
        # Ensure mask is integer type if needed by wandb (usually okay)
        img_np = img_np.astype(np.uint8) # Or int, check wandb requirements if issues persist

    return img_np


class WandBLogger:
    """Handles all WandB interactions in one place"""

    def __init__(self, config: DictConfig, model: torch.nn.Module | None = None, project: str | None = None, run_name: str | None = None):
        """
        Initializes Weights & Biases logging.
        # ... (Args documentation remains the same) ...
        """
        resolved_config = OmegaConf.to_container(config, resolve=True)

        # Use provided args or fallback to config
        _project = project if project else config.tracking.wandb_project
        _name = run_name if run_name else config.tracking.run_name

        try:
            self.run = wandb.init(
                project=_project,
                name=_name,
                config=resolved_config,
                # resume="allow", # Keep commented unless needed
            )
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            self.run = None # Set run to None if initialization fails

        if self.run and model is not None:
            # wandb.watch should ideally respect the defined step metric or operate independently.
            # This setup clarifies your intent for custom logs.
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")

            wandb.watch(model, log='all', log_freq=100) # Log gradients and parameters

    def log_metrics(self, metrics: dict, step: int | None = None, commit: bool = True):
        """Log metrics with optional step parameter."""
        if self.run:
            self.run.log(metrics, step=step, commit=commit) # commit=True ensures data is synced

    def log_weights(self, weights: list[torch.Tensor] | torch.nn.ParameterList, step: int | None = None, commit: bool = True):
        """Log deep supervision weights with optional step parameter."""
        if not self.run:
            return 
            
        # Handle both list of tensors and ParameterList
        if isinstance(weights, torch.nn.ParameterList):
             weight_values = [w.data.item() for w in weights] # Use .data to get tensor
        else:
             weight_values = [w.item() for w in weights]

        metrics = {f"ds_weights/layer_{i}": w for i, w in enumerate(weight_values)}
        self.log_metrics(metrics, step=step, commit=commit)


    
    def log_segmentation_masks(
        self,
        images: torch.Tensor,      # Shape (B, C, D, H, W)
        true_masks: torch.Tensor,  # Shape (B, D, H, W) - Assuming masks are labels
        pred_masks: torch.Tensor,  # Shape (B, D, H, W) - Assuming masks are labels
        step: int | None = None,
        class_labels: dict | None = None,
        max_samples: int = 4,
        slice_dim: int = 2, # Dimension to slice along (0=Batch, 1=Channel(ignore), 2=Depth, 3=Height, 4=Width)
        slice_index: int | None = None, # Index to slice (None for middle)
        commit: bool = True
    ):
        if not self.run:
            return

        try:
            # Detach tensors FIRST
            images = images.cpu().detach()
            true_masks = true_masks.cpu().detach()
            pred_masks = pred_masks.cpu().detach()

            num_samples = min(images.shape[0], max_samples)
            log_dict = {}

            for i in range(num_samples):
                # --- Prepare Slices ---
                # Assuming input image is (B, C, D, H, W), take C=0
                image_vol = images[i, 0] # Shape (D, H, W)
                true_mask_vol = true_masks[i] # Shape (D, H, W)
                pred_mask_vol = pred_masks[i] # Shape (D, H, W)

                # Slice dimension relative to the 3D volume (D, H, W)
                slice_dim_3d = slice_dim - 2 # Adjust if slice_dim referred to 5D tensor

                img_slice_np = _prepare_slice_for_wandb(image_vol, is_mask=False, slice_dim=slice_dim_3d, slice_index=slice_index)
                true_mask_slice_np = _prepare_slice_for_wandb(true_mask_vol, is_mask=True, slice_dim=slice_dim_3d, slice_index=slice_index)
                pred_mask_slice_np = _prepare_slice_for_wandb(pred_mask_vol, is_mask=True, slice_dim=slice_dim_3d, slice_index=slice_index)

                log_key = f"val/segmentation_sample_{i}"
                log_dict[log_key] = wandb.Image(
                    img_slice_np,
                    masks={
                        "predictions": {"mask_data": pred_mask_slice_np, "class_labels": class_labels},
                        "ground_truth": {"mask_data": true_mask_slice_np, "class_labels": class_labels},
                    },
                    caption=f"Sample {i} Slice [:, :, {slice_index if slice_index is not None else 'middle'}]" # Indicate slice
                )
                # --------------------

            self.log_metrics(log_dict, step=step, commit=commit)

        except Exception as e:
            print(f"!!! ERROR in log_segmentation_masks: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            print("!!! Skipping image logging for this step due to error.")


    def finalize(self, exit_code: int | None = None):
        """Finishes the WandB run."""
        if self.run:
            self.run.finish(exit_code=exit_code)
            print("WandB run finished.")

    @property
    def is_active(self) -> bool:
        """Check if the WandB run is active."""
        return self.run is not None

def get_wandb_logger(config: DictConfig, model: torch.nn.Module | None = None) -> WandBLogger | None:
    """
    Initializes and returns a WandBLogger instance based on config.
    Returns None if tracking is disabled in the config.
    """
    if config.tracking.get("log_to_wandb", False): # Check if wandb logging is enabled
        return WandBLogger(config=config, model=model)
    else:
        print("WandB tracking is disabled in the configuration.")
        return None # Return None or a dummy logger if needed