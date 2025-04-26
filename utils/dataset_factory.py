# dataset_factory.py

# TODO: consider using this:
# Usage:
# from utils.dataset_factory import get_dataset
# 
#@register_dataset("my_task")
#class MyDataset(torch.utils.data.Dataset):
#...
# And then use:
#dataset = get_dataset("my_task", cfg, phase)

DATASET_MAPPING = {}

def register_dataset(name):
    def decorator(cls):
        DATASET_MAPPING[name] = cls
        return cls
    return decorator

def get_dataset(task_name, *args, **kwargs):
    if task_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {list(DATASET_MAPPING.keys())}")
    
    DatasetClass = DATASET_MAPPING[task_name]
    return DatasetClass(*args, **kwargs)


