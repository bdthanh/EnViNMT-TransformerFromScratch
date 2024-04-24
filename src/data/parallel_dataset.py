import torch
from typing import Any
from tqdm import tqdm
from torch.utils.data import Dataset
from .tokenizer import BaseTokenizer

class ParallelDataset(Dataset):
  
    def __init__(self, src_dataset, trg_dataset, src_tokenizer: BaseTokenizer, trg_tokenizer: BaseTokenizer, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_tokenized_ds = []
        self.trg_tokenized_ds = []
        for i in range(len(self)):
            src_sentence = self.src_dataset[i]
            trg_sentence = self.trg_dataset[i]
            tokenized_src = self.src_tokenizer.tokenize(src_sentence)
            tokenized_trg = self.trg_tokenizer.tokenize(trg_sentence)
            if len(tokenized_src) < 255 and len(tokenized_trg) < 255: 
                self.src_tokenized_ds.append(tokenized_src)
                self.trg_tokenized_ds.append(tokenized_trg)
        
    def __len__(self):
        assert len(self.src_dataset) == len(self.trg_dataset), 'Source and target dataset must have the same length'
        return len(self.src_dataset)  
    
    def __getitem__(self, index) -> Any:
        src_text = self.src_dataset[index]
        trg_text = self.trg_dataset[index]
        
        enc_input_tokens = self.src_tokenized_ds[index]
        dec_input_tokens = self.trg_tokenized_ds[index]
        
        enc_input_tensor = self.src_tokenizer.vocab.sentence_to_tensor(enc_input_tokens)
        dec_input_tensor = self.trg_tokenizer.vocab.sentence_to_tensor(dec_input_tokens)
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <sos> and <eos>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # We will only add <sos>
      
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
          
        # Add <sos> and <eos> token
        encoder_input = torch.cat(
            [
                torch.tensor([self.src_tokenizer.vocab.sos_id], dtype=torch.int64),
                enc_input_tensor,
                torch.tensor([self.src_tokenizer.vocab.eos_id], dtype=torch.int64),
                torch.tensor([self.src_tokenizer.vocab.pad_id] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <sos> token
        decoder_input = torch.cat(
            [
                torch.tensor([self.trg_tokenizer.vocab.sos_id], dtype=torch.int64),
                dec_input_tensor,
                torch.tensor([self.trg_tokenizer.vocab.pad_id] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <eos> token, label should be the same as decoder_input but shifted by one
        label = torch.cat(
            [
                dec_input_tensor,
                torch.tensor([self.trg_tokenizer.vocab.eos_id], dtype=torch.int64),
                torch.tensor([self.trg_tokenizer.vocab.pad_id] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {
            'src_text': src_text,
            'trg_text': trg_text,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': label,
            'encoder_mask': (encoder_input != self.src_tokenizer.vocab.pad_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.trg_tokenizer.vocab.pad_id).unsqueeze(0).int() & nopeak_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
        }
        
def nopeak_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0