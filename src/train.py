import warnings
warnings.filterwarnings("ignore")
import os
import wandb
from tqdm import tqdm
from pathlib import Path

from model.transformer import Transformer
from data.parallel_dataset import ParallelDataset, nopeak_mask
from data.tokenizer import ViTokenizer, EnTokenizer

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def beam_search_decode():
    pass

def greedy_decode():
    pass

def train(config):
    pass
  
if __name__ == '__main__':
    train()
  