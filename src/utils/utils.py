import os

def create_if_missing_folder(path: str):
    """Create a folder if it does not exist.

    Args:
        path (str): The path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)