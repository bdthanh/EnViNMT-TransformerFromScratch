import torch
from torch import nn
from torch.nn import Module

class LayerNormalization(Module):
    def __init__(self, d_model: int = 512, eps: float = 1e-10):
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter.Parameter(torch.ones(d_model), requires_grad=True)
        self.beta = nn.parameter.Parameter(torch.zeros(d_model), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        
        return self.alpha * norm_x + self.beta
      
        
        
