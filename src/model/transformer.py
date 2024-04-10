from torch import Tensor
from torch.nn import Linear, Module
from decoder import Decoder
from encoder import Encoder

class Transformer(Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, n_layers: int = 6, 
                 dropout: float = 0.1, eps: float = 1e-10, max_seq_len: int = 100) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size=src_vocab_size, d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, 
                               dropout=dropout, eps=eps, max_seq_len=max_seq_len)
        self.decoder = Decoder(vocab_size=trg_vocab_size, d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, 
                               dropout=dropout, eps=eps, max_seq_len=max_seq_len)
        self.linear = Linear(d_model, trg_vocab_size)
        
    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, trg_mask: Tensor): 
        x_enc = self.encoder(src, src_mask)
        x_dec = self.decoder(trg, x_enc, src_mask, trg_mask)
        x = self.linear(x_dec)
        
        return x