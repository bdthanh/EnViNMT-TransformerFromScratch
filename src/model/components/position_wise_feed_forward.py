import torch
from torch.nn import Module, Linear, ReLU, Dropout

class PositionWiseFeedForward(Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout1 = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.relu = ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        
        return self.dropout2(x)
