from math import sqrt
from torch.nn import Module, Embedding

class EmbeddingLayer(Module):
  
    def __init__(self, vocab_size, d_model):
        super().__init__()       
        self.d_model = d_model 
        self.embedding = Embedding(vocab_size, d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        
        return x * sqrt(self.d_model)
    