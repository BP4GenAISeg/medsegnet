import re 

def extract_task_name(path: str) -> str:
    """
    Extract the task name from a file path.
    Args:
        path (str): The file path.
    Returns:
        str: The extracted task name.
    """
    pattern = re.compile(r'(Task\d+_[A-Za-z]+)', re.IGNORECASE)
    match = pattern.search(path)
    return match.group(1).lower() if match else None