import torch
import inspect


def get_all_functions():
  """
  Get all functions defined in this module.

  Returns:
      dict: Dictionary with function names as keys and functions as values.
  """

  return {name.lower():func for name, func in globals().items() if callable(func)}

def get_weights_depth(depth):
  """
  Get the weights for deep supervision outputs based on the depth of the model.

  Args:
      depth (int): Depth of the model.

  Returns:
      list: Weights for each output.
  """

  weights = [pow(2, -index) for index in range(1, depth+1)]
  return map(lambda w: w/sum(weights), weights)

def get_fusion_function(prediction_fusion_mode):
  """
  Get the fusion function by name.

  Args:
      prediction_fusion_mode (str): Name of the fusion function.

  Returns:
      callable: Fusion function.
  """
  assert prediction_fusion_mode in get_all_functions(), f"Unknown prediction mode: {prediction_fusion_mode}"
  return get_all_functions()[prediction_fusion_mode]

def apply_fusion_fn(fn, *args, **kwargs):
  """
  Apply a fusion function to outputs with flexible parameters.
  
  Args:
    fn (callable): The fusion function to apply
    outputs (list): The prediction outputs to fuse
    *args: Additional positional arguments to pass to fn
    **kwargs: Additional keyword arguments to pass to fn
    
  Returns:
    The result of applying fn to outputs with the given parameters
  """
  assert callable(fn), f"Expected a callable function, got {type(fn)}"
  assert fn.__name__ in get_all_functions(), f"Function '{fn.__name__}' is not defined in fusion functions"
  sig = inspect.signature(fn)
  try:
    sig.bind(*args, **kwargs)
  except TypeError as e:
    raise TypeError(f"Invalid arguments for fusion function '{fn.__name__}': {e}")
  return fn(*args, **kwargs)

def only_final(outputs):
    return outputs

def weighted_softmax(outputs, weights):
    """
    Combines multiple outputs using a weighted softmax approach.
    
    Args:
      outputs (list[Tensor]): List of logits tensors with shape [B, C, ...].
      weights (list[float]): Weights for each output.
    
    Returns:
      Tensor: Final predictions with shape [B, ...].
    """
    combined_prob = None

    for output, weight in zip(outputs, weights):
        prob = torch.softmax(output, dim=1)
        combined_prob = (combined_prob or 0) + weight * prob
    
    assert combined_prob != None, "Combined_prob is None"
    final_pred = torch.argmax(combined_prob, dim=1)
    
    return final_pred


