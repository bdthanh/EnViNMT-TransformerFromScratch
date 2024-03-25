import torch
from torch import Tensor
from torch.nn import Module

class PositionalEncoding(Module):
    #TODO: Apply NN positional encoding
    def __init__(self, max_seq_len: int, d_model: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len
        self.positional_encoding = torch.zeros(max_seq_len, d_model)
        self.positional_encoding.requires_grad = False
        pos = torch.arange(0, max_seq_len).unsqueeze(dim=1)
        self.positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        self.positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (torch.arange(0, d_model, 2).float() / d_model)))
        
    def forward(self, x: Tensor):
        _, seq_len = x.size()
        
        return self.positional_encoding[:seq_len, :]
