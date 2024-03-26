from torch.nn import Module, Embedding

class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()        
        self.embed = Embedding(vocab_size, d_model)
        
    def forward(self, x):
        x = self.embed(x)
        
        return x
    