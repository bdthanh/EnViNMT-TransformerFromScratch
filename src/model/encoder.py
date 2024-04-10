from copy import deepcopy
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from components.embedding import Embedding
from components.layer_normalization import LayerNormalization
from components.multi_head_attention import MultiHeadAttention
from components.position_wise_feed_forward import PositionWiseFeedForward
from components.positional_encoding import PositionalEncoding 

class Encoder(Module):
    def __init__(self, vocab_size: int, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, n_layers: int = 6, 
                 dropout: float = 0.1, eps: float = 1e-10, max_seq_len: int = 100) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model, dropout=dropout)
        single_layer = EncoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, eps=eps)
        self.encoder_layers = ModuleList([deepcopy(single_layer) for _ in range(n_layers)])
        
    def forward(self, src: Tensor, mask: Tensor):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
            
        return x  
      
class EncoderLayer(Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, dropout: float = 0.1, 
                 eps: float = 1e-10) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.self_attn_layer_norm = LayerNormalization(d_model=d_model, eps=eps)
        self.self_attn_dropout = Dropout(p=dropout)
        
        self.feed_fwd = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.feed_fwd_layer_norm = LayerNormalization(d_model=d_model, eps=eps)
        self.feed_fwd_dropout = Dropout(p=dropout)
        
    def forward(self, x: Tensor, mask: Tensor):
        _x = self.self_attn(x, x, x, mask)
        x = self.self_attn_layer_norm(x + self.self_attn_dropout(_x))
        
        _x = self.feed_fwd(x)
        x = self.feed_fwd_layer_norm(x + self.feed_fwd_dropout(_x))
        
        return x 
        