from copy import deepcopy
from torch import Tensor
from torch.nn import Linear, Dropout, Module, ModuleList
from .components.embedding import EmbeddingLayer
from .components.layer_normalization import LayerNormalization
from .components.multi_head_attention import MultiHeadAttention
from .components.position_wise_feed_forward import PositionWiseFeedForward
from .components.positional_encoding import PositionalEncoding 

class Decoder(Module):
  
    def __init__(self, vocab_size: int, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, n_layers: int = 6, 
                 dropout: float = 0.1, eps: float = 1e-10, max_seq_len: int = 256) -> None:
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model, dropout=dropout)
        single_layer = DecoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, eps=eps)
        self.decoder_layers = ModuleList([deepcopy(single_layer) for _ in range(n_layers)])
        self.linear = Linear(d_model)
        
    def forward(self, trg: Tensor, x_enc: Tensor, src_mask: Tensor, trg_mask: Tensor):
        x = self.embedding(trg)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, x_enc, src_mask, trg_mask)
        x = self.linear(x)
        
        return x 
      
class DecoderLayer(Module):
  
    def __init__(self, d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, 
                 dropout: float = 0.1, eps: float = 1e-10) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.masked_self_attn_layer_norm = LayerNormalization(d_model=d_model, eps=eps)
        self.masked_self_attn_dropout = Dropout(p=dropout)
        
        self.enc_dec_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.enc_dec_attn_layer_norm = LayerNormalization(d_model=d_model, eps=eps)
        self.enc_dec_attn_dropout = Dropout(p=dropout)
        
        self.feed_fwd = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.feed_fwd_layer_norm = LayerNormalization(d_model=d_model, eps=eps)
        self.feed_fwd_dropout = Dropout(p=dropout)
        
    def forward(self, x: Tensor, x_enc: Tensor, src_mask: Tensor, trg_mask: Tensor):
        _x = self.masked_self_attn(x, x, x, trg_mask)
        x = self.masked_self_attn_layer_norm(x+ self.masked_self_attn_dropout(_x))
        
        if x_enc is not None:
            _x = self.masked_self_attn(q=x, k=x_enc, v=x_enc, mask=src_mask)
            x = self.masked_self_attn_layer_norm(x + self.masked_self_attn_dropout(_x))
        
        _x = self.feed_fwd(x)
        x = self.feed_fwd_layer_norm(x + self.feed_fwd_dropout(_x))
        
        return x         
        