from tabulate import tabulate
import numpy as np

def print_table(data, headers=("Property", "Value"), tablefmt="pretty"):
    """
    Prints a dictionary or list of (key, value) pairs in tabular format.

    Args:
        data: A dictionary {key: value} or a list of [key, value].
        headers (tuple): Column headers.
        tablefmt (str): A valid tabulate table format.
    """
    # If data is a dictionary, convert to list of [key, value].
    if isinstance(data, dict):
        data = [[k, v] for k, v in data.items()]
    # If data is already a list of pairs, we assume it's fine.

    print(tabulate(data, headers=headers, tablefmt=tablefmt))


def print_norm_image_stats(normalized_image):
    stats = {
        "Shape": normalized_image.shape,
        "Min pixel value": np.min(normalized_image),
        "Max pixel value": np.max(normalized_image),
        "Mean pixel value": np.mean(normalized_image),
        "Std. Deviation": np.std(normalized_image),
        "Variance": np.var(normalized_image)
    }
    print_table(stats)

  