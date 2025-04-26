import torch
import inspect
from typing import Dict, Callable, Any, List, Sequence, Optional
import torch.nn.functional as F
import logging # Import logging

# Get a logger for this module
logger = logging.getLogger(__name__)

class OutputFuser:
    """
    Handles the fusion of multiple model outputs (logits) from different resolutions.
    """
    VALID_MODES = ['weighted_softmax', 'weighted_majority', 'only_final']

    def __init__(self, mode: str, weights: Optional[Sequence[float]] = None):
        """
        Initializes the OutputFuser.

        Args:
            mode (str): The fusion strategy to use. Must be one of VALID_MODES.
            weights (Optional[Sequence[float]]): A sequence of weights corresponding to the outputs.
                                                 Required for 'weighted_softmax' and 'weighted_majority'.
                                                 The number of weights must match the number of outputs passed to fuse().
        """
        _mode_lower = mode.lower()
        if _mode_lower not in self.VALID_MODES:
            error_message = f"Invalid fusion mode '{mode}'. Valid modes are: {self.VALID_MODES}"
            logger.critical(error_message)
            raise ValueError(error_message)

        self.mode = _mode_lower

        if self.mode in ['weighted_softmax', 'weighted_majority']:
            if weights is None:
                raise ValueError(f"Weights must be provided for fusion mode '{self.mode}'")
            if not (isinstance(weights, Sequence) and not isinstance(weights, str)):
                raise TypeError("Weights must be a sequence (e.g., list or tuple).")
            if not all(isinstance(w, (int, float)) for w in weights):
                raise TypeError("All elements in weights must be numbers.")
            
            _weight_sum = sum(weights)
            if _weight_sum <= 0:
                raise ValueError("Sum of weights must be positive.")
            
            # Normalize weights if they don't sum to approximately 1
            if not torch.isclose(torch.tensor(_weight_sum), torch.tensor(1.0)):
                # Replace print with logger.warning
                logger.error(f"Weights provided {list(weights)} do not sum to 1. Normalizing.")
                raise ValueError("Weights must sum to 1.0. Normalizing weights.")
            else:
                self.weights = list(weights) if weights is not None else []  
        elif self.mode == 'only_final':
            self.weights = list()
        else:
            # This case should not be reachable due to __init__ check, but added for safety
            error_message = f"Internal error: Unhandled fusion mode '{self.mode}'"
            logger.critical(error_message)
            raise RuntimeError(error_message)
        
    def fuse(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Fuses the given sequence of output tensors using the configured mode.

        Args:
            outputs (Sequence[torch.Tensor]): A sequence of output tensors (logits)
                                              Shape of each tensor: [B, C, D, H, W].
                                              The order matters, typically from highest resolution (final)
                                              to lowest resolution.

        Returns:
            torch.Tensor: The fused output tensor containing class predictions.
                          Shape: [B, D, H, W].
        """
        is_valid_outputs = isinstance(outputs, Sequence) and not isinstance(outputs, str) and len(outputs) > 0
        if not is_valid_outputs:
             raise ValueError("Outputs must be a non-empty sequence (e.g., list or tuple) of tensors.")

        are_all_tensors = all(isinstance(t, torch.Tensor) for t in outputs)
        if not are_all_tensors:
             raise TypeError("All elements in outputs must be torch.Tensor.")

        is_weight_length_matching = len(outputs) == len(self.weights)
        if not is_weight_length_matching:
            raise ValueError(f"Number of outputs ({len(outputs)}) must match number of weights ({len(self.weights)}) for mode '{self.mode}'")

        if self.mode == 'weighted_softmax':
            return self._fuse_weighted_softmax(outputs, self.weights) 
        elif self.mode == 'weighted_majority':
            return self._fuse_weighted_majority(outputs, self.weights)
        elif self.mode == 'only_final':
            return self._fuse_only_final(outputs)
        else:
            # This case should not be reachable due to __init__ check, but added for safety
            raise RuntimeError(f"Internal error: Unhandled fusion mode '{self.mode}'")

    def _fuse_weighted_softmax(self, outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
        """Applies weighted softmax fusion."""
        combined_prob: Optional[torch.Tensor] = None
        target_shape = outputs[0].shape[2:] # Use shape of the highest resolution output

        for output, weight in zip(outputs, weights):
            # Ensure output matches the target spatial dimensions (D, H, W)
            if output.shape[2:] != target_shape:
                output = F.interpolate(output, size=target_shape, mode='trilinear', align_corners=False)

            prob = torch.softmax(output, dim=1)
            if combined_prob is None:
                combined_prob = weight * prob
            else:
                combined_prob += weight * prob

        if combined_prob is None: 
            raise ValueError("Combined probability is None. Check outputs.")
        
        final_pred = torch.argmax(combined_prob, dim=1)
        return final_pred

    def _fuse_weighted_majority(self, outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
        """Applies weighted majority voting fusion."""
        if not len(outputs) > 0:
             raise ValueError("Outputs cannot be empty for weighted majority.")
        B, C = outputs[0].shape[:2]
        target_shape = outputs[0].shape[2:] # Use shape of the highest resolution output
        device = outputs[0].device
        dtype = outputs[0].dtype

        combined_scores = torch.zeros((B, C) + target_shape, dtype=dtype, device=device)

        for output, weight in zip(outputs, weights):
             # Ensure output matches the target spatial dimensions (D, H, W)
            if output.shape[2:] != target_shape:
                 output = F.interpolate(output, size=target_shape, mode='trilinear', align_corners=False)

            preds = torch.argmax(output, dim=1)  # (B, D, H, W)
            preds_onehot = F.one_hot(preds, num_classes=C).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
            combined_scores += weight * preds_onehot

        final_pred = torch.argmax(combined_scores, dim=1)  # (B, D, H, W)
        return final_pred

    def _fuse_only_final(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Returns predictions from the first (highest resolution) output."""
        if not len(outputs) > 0:
             raise ValueError("Outputs cannot be empty for only_final mode.")
        final_output = outputs[0]
        if final_output.ndim != 5:
             raise ValueError(f"Expected 5D tensor for the final output, got {final_output.ndim}D tensor.")

        probs = torch.softmax(final_output, dim=1) # probs: (B, C, D, H, W)
        preds = torch.argmax(probs, dim=1) # preds: (B, D, H, W)
        return preds
    
    def __call__(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Enables the OutputFuser instance to be called like a function.

        Args:
            outputs (Sequence[torch.Tensor]): A sequence of output tensors (logits).

        Returns:
            torch.Tensor: The fused output tensor containing class predictions.
        """
        return self.fuse(outputs)
    
    
    
    
    
    
    
    
    
    
    
    
#--------------- OLD SHIT


# FUSION_FUNCTIONS = {}

# def register_fusion_function(fn):
#     """Decorator to register a fusion function."""
#     FUSION_FUNCTIONS[fn.__name__.lower()] = fn
#     return fn

   
# def get_fusion_fn(inference_fusion_mode:str) -> Callable[..., torch.Tensor]:
#   """
#   Get the fusion function by name.

#   Args:
#       prediction_fusion_mode (str): Name of the fusion function.

#   Returns:
#       callable: Fusion function.
#   """
  
#   key = inference_fusion_mode.lower()
  
#   assert key in FUSION_FUNCTIONS, f"Unknown prediction fusion mode: {key}. Available modes: {list(FUSION_FUNCTIONS.keys())}"

#   return FUSION_FUNCTIONS[key]

# def call_fusion_fn(fusion_fn: Callable[..., torch.Tensor], **kwargs) -> torch.Tensor:
#     """
#     Calls a fusion function with only the keyword arguments it expects.

#     Args:
#         fusion_fn (Callable): The fusion function to call.
#         **kwargs: A dictionary of potential arguments (e.g., outputs, weights, extra_arg, etc.).

#     Returns:
#         torch.Tensor: The result of the fusion function.
#     """
#     sig = inspect.signature(fusion_fn)
#     valid_args = {name: kwargs[name] for name in sig.parameters if name in kwargs}
#     try:
#         sig.bind(**valid_args)
#     except TypeError as e:
#         raise TypeError(f"Failed to bind arguments for fusion function '{fusion_fn.__name__}': {e}")
#     return fusion_fn(**valid_args)

# @register_fusion_function
# def only_final(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
#     """
#     Returns only the final output from the model.

#     Args:
#       outputs Tuple[Tensor]: Tuple of tensors where the first tensor is the final output.

#     Returns:
#       Tensor: Final predictions with shape [B, ...].
#     """
#     assert len(outputs) == 1, "Only one output is expected for the 'only_final' fusion mode."
    
#     output = outputs[0]
#     assert output.ndim == 5, f"Expected 5D tensor for the final output, got {output.ndim}D tensor."     

#     probs = torch.softmax(output, dim=1) # probs: (B, C, D, H, W)
#     preds = torch.argmax(probs, dim=1) # preds: (B, D, H, W)

#     return preds

# @register_fusion_function
# def weighted_softmax(outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
#     """
#     Combines multiple outputs using a weighted softmax approach.
    
#     Args:
#       outputs (Sequence[torch.Tensor]): Sequence of logits tensors with shape [B, C, ...].
#       weights (Sequence[float]): Weights for each output.
    
#     Returns:
#       torch.Tensor: Final predictions with shape [B, ...].
#     """
#     combined_prob: torch.Tensor | None = None 
#     for output, weight in zip(outputs, weights):
#         prob = torch.softmax(output, dim=1)
#         if combined_prob is None:
#             combined_prob = weight * prob
#         else:
#             combined_prob = combined_prob + weight * prob

#     if combined_prob is None:
#         raise ValueError("Combined_prob is None. Check that outputs is not empty.")
    
#     final_pred = torch.argmax(combined_prob, dim=1)
#     return final_pred


# @register_fusion_function
# def weighted_majority(outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
#     """
#     Weighted Majority Voting Fusion.
    
#     Each output is converted to class predictions, one-hot encoded,
#     multiplied by its corresponding weight, then summed.
#     Final prediction is argmax over the class axis.

#     Args:
#         outputs (Sequence[Tensor]): List of tensors of shape [B, C, D, H, W].
#         weights (Sequence[float]): Corresponding weights for each output.

#     Returns:
#         Tensor: Final prediction map of shape [B, D, H, W].
#     """
#     assert len(outputs) == len(weights), "Mismatch between number of outputs and weights"

#     B, C, D, H, W = outputs[0].shape
#     combined_scores = torch.zeros((B, C, D, H, W), dtype=outputs[0].dtype, device=outputs[0].device)

#     for output, weight in zip(outputs, weights):
        
#         preds = torch.argmax(output, dim=1)  # (B, D, H, W)
#         preds_onehot = F.one_hot(preds, num_classes=C).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
#         combined_scores += weight * preds_onehot

#     final_pred = torch.argmax(combined_scores, dim=1)  # (B, D, H, W)
#     return final_pred



