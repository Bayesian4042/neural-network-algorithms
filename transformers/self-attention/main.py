from torch import nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qvk_bias: bool = True):
        super().__init__()
        self.d_out = d_out
        
        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bias)
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
