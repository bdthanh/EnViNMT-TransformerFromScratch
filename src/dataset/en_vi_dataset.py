import torch
from typing import Any
from torch.utils.data import Dataset

class EnViDataset(Dataset):
    def __init__(self, src_dataset, trg_dataset, src_tokenizer, trg_tokenizer, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.sos_token = torch.tensor([trg_tokenizer.token_to_id('<sos>')], dtype=torch.int64)
        self.eos_token = torch.tensor([trg_tokenizer.token_to_id('<eos>')], dtype=torch.int64)
        self.pad_token = torch.tensor([trg_tokenizer.token_to_id('<pad>')], dtype=torch.int64)
        self.unk_token = torch.tensor([trg_tokenizer.token_to_id('<unk>')], dtype=torch.int64)
    
    def __len__(self):
        assert len(self.src_dataset) == len(self.trg_dataset), 'Source and target dataset must have the same length'
        return len(self.src_dataset)  
    
    def __getitem__(self, index) -> Any:
        src_text = self.src_dataset[index]
        trg_text = self.trg_dataset[index]
        
        enc_input_tokens = self.src_tokenizer.encode(src_text)
        dec_input_tokens = self.trg_tokenizer.encode(trg_text)
        
        # Add sos, eos and pad to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <sos> and <eos>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # We will only add <sos>
      
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
          
        # Add <sos> and <eos> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <sos> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <eos> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {
            'src_text': src_text,
            'trg_text': trg_text,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'label': label,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & nopeak_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
        }
        
def nopeak_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0