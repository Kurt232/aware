"""
Aware
Jan 3 2025: wjdu
"""

import torch
from torch import nn
from models.units import Mlp, SeqAttention

class AwareLayer(nn.Module):
    def __init__(self, clip_dim, d_model, n_head, mlp_ratio=4, dropout=0):
        super().__init__()
        self.att = SeqAttention(d_model, n_head, proj_drop=dropout)
        self.prior_mlp = Mlp(in_features=clip_dim, out_features=d_model, hidden_features=int(d_model*mlp_ratio), act_layer=nn.GELU, drop=dropout)
        self.mlp = Mlp(in_features=d_model, out_features=d_model, hidden_features=int(d_model*mlp_ratio), act_layer=nn.GELU, drop=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, prior_emb):
        # Project prior embedding
        emb = self.prior_mlp(prior_emb)  # [B, 1, D]
        
        # Reshape inputs
        B, V, L, D = x.shape
        x = x.view(B*V, L, D)  # [B*V, L, D]
        
        # Expand and append prior embedding to sequence
        emb = emb.unsqueeze(1).expand(B, V, 1, D).reshape(B*V, 1, D)  # [B*V, 1, D]
        x_with_emb = torch.cat([x, emb], dim=1)  # [B*V, L+1, D]
        
        # Apply self-attention
        x_out = self.att(x_with_emb)  # [B*V, L+1, D]
        
        # Split back to original sequence and prior embedding
        x = x_out[:, :-1]  # [B*V, L, D]
        x = x.view(B, V, L, D)
        x = self.mlp(x)
        return self.norm(x)
