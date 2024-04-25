import os
import yaml
import torch 

def choose_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def create_if_missing_folder(path: str):
    """Create a folder if it does not exist.

    Args:
        path (str): The path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def is_file_exist(path: str):
    return os.path.exists(path)
        
def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
  