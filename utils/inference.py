import torch
import inspect
from typing import Dict, Callable, Any, List, Sequence
import torch.nn.functional as F

FUSION_FUNCTIONS = {}

def register_fusion_function(fn):
    """Decorator to register a fusion function."""
    FUSION_FUNCTIONS[fn.__name__.lower()] = fn
    return fn

def compute_weights_depth(depth:int) -> List[float]:
  """
  Get the weights for deep supervision outputs based on the depth of the model.

  Args:
      depth (int): Depth of the model.

  Returns:
      list: Weights for each output.
  """

  weights = [pow(2, -index) for index in range(1, depth+1)]
  total = sum(weights)
  return list(map(lambda w: w / total, weights))
   
def get_fusion_fn(inference_fusion_mode:str) -> Callable[..., torch.Tensor]:
  """
  Get the fusion function by name.

  Args:
      prediction_fusion_mode (str): Name of the fusion function.

  Returns:
      callable: Fusion function.
  """
  
  key = inference_fusion_mode.lower()
  
  assert key in FUSION_FUNCTIONS, f"Unknown prediction fusion mode: {key}. Available modes: {list(FUSION_FUNCTIONS.keys())}"

  return FUSION_FUNCTIONS[key]

def call_fusion_fn(fusion_fn: Callable[..., torch.Tensor], **kwargs) -> torch.Tensor:
    """
    Calls a fusion function with only the keyword arguments it expects.

    Args:
        fusion_fn (Callable): The fusion function to call.
        **kwargs: A dictionary of potential arguments (e.g., outputs, weights, extra_arg, etc.).

    Returns:
        torch.Tensor: The result of the fusion function.
    """
    sig = inspect.signature(fusion_fn)
    valid_args = {name: kwargs[name] for name in sig.parameters if name in kwargs}
    try:
        sig.bind(**valid_args)
    except TypeError as e:
        raise TypeError(f"Failed to bind arguments for fusion function '{fusion_fn.__name__}': {e}")
    return fusion_fn(**valid_args)

@register_fusion_function
def only_final(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Returns only the final output from the model.

    Args:
      outputs Tuple[Tensor]: Tuple of tensors where the first tensor is the final output.

    Returns:
      Tensor: Final predictions with shape [B, ...].
    """
    assert len(outputs) == 1, "Only one output is expected for the 'only_final' fusion mode."
    
    output = outputs[0]
    assert output.ndim == 5, f"Expected 5D tensor for the final output, got {output.ndim}D tensor."     

    probs = torch.softmax(output, dim=1) # probs: (B, C, D, H, W)
    preds = torch.argmax(probs, dim=1) # preds: (B, D, H, W)

    return preds

@register_fusion_function
def weighted_softmax(outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """
    Combines multiple outputs using a weighted softmax approach.
    
    Args:
      outputs (Sequence[torch.Tensor]): Sequence of logits tensors with shape [B, C, ...].
      weights (Sequence[float]): Weights for each output.
    
    Returns:
      torch.Tensor: Final predictions with shape [B, ...].
    """
    combined_prob: torch.Tensor | None = None 
    for output, weight in zip(outputs, weights):
        prob = torch.softmax(output, dim=1)
        if combined_prob is None:
            combined_prob = weight * prob
        else:
            combined_prob = combined_prob + weight * prob

    if combined_prob is None:
        raise ValueError("Combined_prob is None. Check that outputs is not empty.")
    
    final_pred = torch.argmax(combined_prob, dim=1)
    return final_pred


@register_fusion_function
def weighted_majority(outputs: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """
    Weighted Majority Voting Fusion.
    
    Each output is converted to class predictions, one-hot encoded,
    multiplied by its corresponding weight, then summed.
    Final prediction is argmax over the class axis.

    Args:
        outputs (Sequence[Tensor]): List of tensors of shape [B, C, D, H, W].
        weights (Sequence[float]): Corresponding weights for each output.

    Returns:
        Tensor: Final prediction map of shape [B, D, H, W].
    """
    assert len(outputs) == len(weights), "Mismatch between number of outputs and weights"

    B, C, D, H, W = outputs[0].shape
    combined_scores = torch.zeros((B, C, D, H, W), dtype=outputs[0].dtype, device=outputs[0].device)

    for output, weight in zip(outputs, weights):
        
        preds = torch.argmax(output, dim=1)  # (B, D, H, W)
        preds_onehot = F.one_hot(preds, num_classes=C).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        combined_scores += weight * preds_onehot

    final_pred = torch.argmax(combined_scores, dim=1)  # (B, D, H, W)
    return final_pred



