import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import torch
from tqdm import tqdm
from pathlib import Path

from data.utils import load_data
from model.transformer import Transformer
from data.parallel_dataset import ParallelDataset, nopeak_mask
from data.tokenizer import ViTokenizer, EnTokenizer

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def choose_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ds(config):
    train_src_dataset = load_data(path=config['data']['train']['src'], lowercase=True)
    train_trg_dataset = load_data(path=config['data']['train']['trg'], lowercase=True)
    valid_src_dataset = load_data(path=config['data']['valid']['src'], lowercase=True)
    valid_trg_dataset = load_data(path=config['data']['valid']['trg'], lowercase=True)

    src_tokenizer = EnTokenizer()
    trg_tokenizer = ViTokenizer()

    src_tokenizer.build_vocab(train_src_dataset, is_tokenized=False, min_freq=config['min_freq'])
    trg_tokenizer.build_vocab(train_trg_dataset, is_tokenized=False, min_freq=config['min_freq'])
    
    train_ds = ParallelDataset(train_src_dataset, train_trg_dataset, src_tokenizer, trg_tokenizer, config['seq_len'])
    val_ds = ParallelDataset(valid_src_dataset, valid_trg_dataset, src_tokenizer, trg_tokenizer, config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    assert len(train_src_dataset) == len(train_trg_dataset), 'Source and target dataset must have the same length'
    
    for i in range(len(train_src_dataset)):
        src_ids = src_tokenizer.tokenize(train_src_dataset[i])
        tgt_ids = trg_tokenizer.tokenize(train_trg_dataset[i])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer

def train(config):
    device = choose_device()
    
    pass
  
if __name__ == '__main__':
    train()
  