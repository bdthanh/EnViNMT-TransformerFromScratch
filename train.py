import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import torch
import torchmetrics
from tqdm import tqdm
from pathlib import Path

from src.data.utils import load_data
from src.data.tokenizer import BaseTokenizer, EnTokenizer, ViTokenizer
from src.data.parallel_dataset import ParallelDataset, nopeak_mask 
from src.utils import create_if_missing_folder, load_config, is_file_exist
from src.model.transformer import Transformer, get_model
from src.output_decode import beam_search_decode

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss

def choose_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    return torch.device(device)
  
def save_checkpoint(path, model, optimizer, epoch, global_step):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    
def load_checkpoint_if_exists(path, model, optimizer):
    initial_epoch, global_step = 0, 0
    if is_file_exist(path):
        checkpoint = torch.load(config['checkpoint_last'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f'Loaded checkpoint from epoch {initial_epoch}')
    return model, optimizer, initial_epoch, global_step

def get_ds(config):
    print(f'Loading dataset...')
    train_src_dataset = load_data(path=config['train_src'], lowercase=True) # max_len = 786
    train_trg_dataset = load_data(path=config['train_trg'], lowercase=True) # max_len = 788
    valid_src_dataset = load_data(path=config['valid_src'], lowercase=True)
    valid_trg_dataset = load_data(path=config['valid_trg'], lowercase=True)

    src_tokenizer = EnTokenizer(config['vocab_en'])
    trg_tokenizer = ViTokenizer(config['vocab_vi'])
    
    if not src_tokenizer.vocab_exist:
        src_tokenizer.build_vocab(train_src_dataset, is_tokenized=False, min_freq=config['min_freq'])
    if not trg_tokenizer.vocab_exist:    
        trg_tokenizer.build_vocab(train_trg_dataset, is_tokenized=False, min_freq=config['min_freq'])
    
    train_ds = ParallelDataset(train_src_dataset, train_trg_dataset, src_tokenizer, trg_tokenizer, config['max_seq_len'])
    val_ds = ParallelDataset(valid_src_dataset, valid_trg_dataset, src_tokenizer, trg_tokenizer, config['max_seq_len'])
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer

def epoch_eval(model: Transformer, loss: CrossEntropyLoss, global_step: int, epoch: int, val_dataloader: DataLoader, 
               enc_mask, src_tokenizer: BaseTokenizer, trg_tokenizer: BaseTokenizer, max_seq_len, device):
    model.eval()
    sos_id = trg_tokenizer.vocab.sos_id
    eos_id = trg_tokenizer.vocab.eos_id
    target_list, pred_list = [], []
    epoch_loss = 0  
    batch_iter = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}", total=len(val_dataloader))
    with torch.no_grad():
        for batch in batch_iter:
            enc_input = batch['encoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            src_text = batch['src_text'][0]
            trg_text = batch['trg_text'][0]
            label = batch['label'].to(device)
            enc_output = model.encoder(enc_input, enc_mask)
            dec_input = torch.full((1, 1), sos_id, dtype=enc_input.dtype, device=device)
            next_token = ''
            
            #TODO: Implement beam search instead of greedy search
            while dec_input.size(1) != max_seq_len+1 and next_token != eos_id:
                dec_mask = nopeak_mask(dec_input.size(1)).type_as(enc_mask).to(device)
                dec_output = model.decoder(dec_input, enc_output, enc_mask, dec_mask)
                prob = model.linear(dec_output[:, -1]) # [:, -1] for the last token
                _, next_token = torch.max(prob, dim=1)
                dec_input = torch.cat([
                    dec_input, torch.full((1, 1), next_token.item(), dtype=enc_input.dtype, device=device)
                ], dim=1)
            pred_sent = trg_tokenizer.tensor_to_sentence(dec_input[0, 1:-1]) # remove sos and eos tokens
            target_list.append(trg_text)
            pred_list.append(pred_sent)
            print(f"Source: {src_text}")
            print(f"Target: {trg_text}")
            print(f"Predicted: {pred_sent}")
            
    char_error_rate = torchmetrics.CharErrorRate()
    cer_score = char_error_rate(pred_list, target_list)
    wandb.log({'validation/cer': cer_score, 'global_step': global_step})

    word_error_rate = torchmetrics.WordErrorRate()
    wer_score = word_error_rate(pred_list, target_list)
    wandb.log({'validation/wer': wer_score, 'global_step': global_step})

    bleu = torchmetrics.BLEUScore()
    bleu_score = bleu(pred_list, target_list)
    wandb.log({'validation/BLEU': bleu_score, 'global_step': global_step})            

def train(config):
    device = choose_device()
    train_dataloader, val_dataloader, src_tokenizer, trg_tokenizer = get_ds(config)
    model = get_model(config=config, src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer).to(device)
    optimizer = Adam(model.parameters(), lr=config['init_lr'], eps= 1e-10)
    loss_func = CrossEntropyLoss(ignore_index=src_tokenizer.vocab.pad_id, label_smoothing=0.1).to(device)    
    max_seq_len = config['max_seq_len']
    initial_epoch, global_step = 0, 0 
    model, optimizer, initial_epoch, global_step = load_checkpoint_if_exists(config['checkpoint_last'], model, optimizer)
    best_loss = float('inf')
    
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    
    print(f'_________ START TRAINING __________')
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
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            batch_iter.set_postfix_str(f"Loss: {loss.item():.6f}")
            optimizer.step()
            global_step += 1

        epoch_eval(model, loss_func, global_step, epoch, val_dataloader, enc_mask, src_tokenizer, trg_tokenizer, max_seq_len, device)
        save_checkpoint(config['checkpoint_last'], model, optimizer, epoch, global_step)
    print(f'_________ END TRAINING __________')
  
if __name__ == '__main__':
    current_file_path = Path(__file__).resolve() 
    current_dir = current_file_path.parent 
    config_path = current_dir / 'config.yaml'
    config = load_config(config_path)
    create_if_missing_folder(config['checkpoint_dir'])
    create_if_missing_folder(config['vocab_dir'])
    wandb.init(project='en_vi_nmt', config=config)
    train(config)
    wandb.finish()
  