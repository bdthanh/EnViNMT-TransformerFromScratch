import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import torch
from tqdm import tqdm
from pathlib import Path

from data.utils import load_data
from data.tokenizer import BaseTokenizer
from utils import create_if_missing_folder, load_config
from model.transformer import Transformer, get_model
from data.parallel_dataset import ParallelDataset, nopeak_mask
from data.tokenizer import ViTokenizer, EnTokenizer

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss

def choose_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ds(config):
    train_src_dataset = load_data(path=config['train_src'], lowercase=True)
    train_trg_dataset = load_data(path=config['train_trg'], lowercase=True)
    valid_src_dataset = load_data(path=config['valid_src'], lowercase=True)
    valid_trg_dataset = load_data(path=config['valid_trg'], lowercase=True)

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

def epoch_eval(model: Transformer, val_dataloader, enc_mask, src_tokenizer: BaseTokenizer, trg_tokenizer: BaseTokenizer, max_seq_len, device):
    model.eval()
    sos_id = trg_tokenizer.vocab.sos_id
    eos_id = trg_tokenizer.vocab.eos_id
    with torch.no_grad():
        for batch in val_dataloader:
            enc_input = batch['encoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            src_text = batch['src_text'][0]
            trg_text = batch['trg_text'][0]

            enc_output = model.encoder(enc_input, enc_mask)
            dec_input = torch.full((1, 1), sos_id, dtype=enc_input.dtype, device=device)
            next_token = ''
            while dec_input.size(1) != max_seq_len and next_token != eos_id:
                dec_mask = nopeak_mask(dec_input.size(1)).type_as(enc_mask).to(device)
                dec_output = model.decoder(dec_input, enc_output, enc_mask, dec_mask)
                prob = model.linear(dec_output[:, -1]) 
                _, next_token = torch.max(prob, dim=1)
                dec_input = torch.cat([
                    dec_input, torch.full((1, 1), next_token.item(), dtype=enc_input.dtype, device=device)
                ], dim=1)
    return dec_input.squeeze(0)

def train(config):
    device = choose_device()
    train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer = get_ds(config)
    model = get_model(config=config, src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    optimizer = Adam(model.parameters(), lr=config['init_lr'], eps= 1e-10)
    loss_func = CrossEntropyLoss(ignore_index=src_tokenizer.vocab.pad_id, label_smoothing=0.1).to(device)    
    max_seq_len = config['max_seq_len']
    initial_epoch, global_step = 0, 0 
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", total=len(train_dataloader))
        for batch in batch_iter:
            optimizer.zero_grad()
            enc_input = batch['encoder_input'].to(device)
            dec_input = batch['decoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            output = model(src=enc_input, trg=dec_input, src_mask=enc_mask, trg_mask=dec_mask)
            
            loss = loss_func(output.transpose(1, 2), label)
            loss.backward()
            batch_iter.set_postfix_str(f"Loss: {loss.item():.6f}")
            optimizer.step()
            global_step += 1

        epoch_eval(model, val_dataloader, enc_mask, src_tokenizer, trg_tokenizer, max_seq_len, device)
  
if __name__ == '__main__':
    config = load_config()
    create_if_missing_folder(config['checkpoint_dir'])
    create_if_missing_folder(config['vocab_dir'])
    wandb.init(project='en_vi_nmt', config=config)
    train(config)
  