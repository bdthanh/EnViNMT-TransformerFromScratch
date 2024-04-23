import torch
from torch import Tensor
from torch.nn import Dropout, Module

class PositionalEncoding(Module):
  
    def __init__(self, max_seq_len: int, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len
        positional_encoding = torch.zeros(max_seq_len, d_model)
        positional_encoding.requires_grad = False
        pos = torch.arange(0, max_seq_len).unsqueeze(dim=1)
        positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        positional_encoding = positional_encoding.unsqueeze(0)  
        self.register_buffer('positional_encoding', positional_encoding)
        self.dropout = Dropout(p=dropout)
        
    def forward(self, x: Tensor):
        _, seq_len, _ = x.size() # [batch_size, seq_len, d_model]
        x = x + self.positional_encoding[:, :seq_len]
        return self.dropout(x)
