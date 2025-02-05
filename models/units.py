"""
UniTS
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from timm.models.layers import DropPath
from timm.models.layers.helpers import to_2tuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, query):
        B, N, C = x.shape
        q = self.q(query).reshape(
            B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        var_num = query.shape[1]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DynamicLinear(nn.Module):
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)


class DynamicLinearMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv1d(in_features, hidden_features,
                             3, groups=group, bias=bias[0], padding=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.seq_fc = DynamicLinear(
            hidden_features//4, hidden_features//4, bias=bias[1], fixed_in=prefix_token_length)
        self.prompt_fc = DynamicLinear(
            hidden_features//4, prefix_token_length, bias=bias[1], fixed_in=prefix_token_length)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def dynamic_linear(self, x, prefix_seq_len):
        x_func = x[:, :, prefix_seq_len:]
        x_seq = x[:, :, :prefix_seq_len]
        x_seq_out = self.seq_fc(
            x_seq, x_seq.shape[-1]-self.prefix_token_length)
        x_prompt = self.prompt_fc(x_seq, self.prefix_token_length)
        x = torch.cat((x_prompt, x_seq_out, x_func), dim=-1)
        return x

    def split_dynamic_linear(self, x, prefix_seq_len):
        x1, x2 = x.chunk(2, dim=-2)
        x1 = self.dynamic_linear(x1, prefix_seq_len)
        return torch.cat((x1, x2), dim=-2)

    def forward(self, x, prefix_seq_len, dim=2):
        n, var, l, c = x.shape
        x = x.view(-1, l, c)
        x = x.transpose(-1, -2)
        x = self.fc1(x)
        x = self.split_dynamic_linear(x, prefix_seq_len)
        x = self.act(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(x).view(n, var, l, c)
        x = self.drop2(x)
        return x


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset+x.size(2)]


class SeqAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5) # [3, B, P, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0) # [B, P, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False) # [B, num_heads, N, head_dim]
        k = k.mean(dim=1, keepdim=False) # [B, num_heads, N, head_dim]
        # [B, num_heads, N, head_dim, P] -> [B, num_heads, N, head_dim * P]
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(self.ls1(x))
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn_var(self.norm1(x))))
        return x


class CtxAttBlock(nn.Module):
    def __init__(self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads,
            qkv_bias,
            qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
    
    def forward(self, x, ctx):
        B, V, L, C = x.shape
        x = x.view(-1, L, C) # [B*V, L, C]
        ctx = ctx.unsqueeze(1) # [B, 1, N, C]
        ctx = ctx.repeat(1, V, 1, 1).view(-1, ctx.shape[-2], ctx.shape[-1]) # [B*V, N, C]
        cross_att_out = self.cross_attn(ctx, x) # [B*V, L, C]
        x = x + cross_att_out
        x = x.view(B, V, L, C)
        return self.norm1(x)


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is DynamicLinearMlp:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.ctx_att_block = CtxAttBlock(dim=dim, num_heads=num_heads,
                                         attn_drop=attn_drop, proj_drop=proj_drop,
                                         norm_layer=norm_layer)

        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, mlp_layer=Mlp,
                                    proj_drop=proj_drop, init_values=init_values, drop_path=drop_path,
                                    act_layer=act_layer, norm_layer=norm_layer,
                                    )

    def forward(self, x, ctx=None):
        x = self.seq_att_block(x, attn_mask=None)
        x = self.var_att_block(x)
        if ctx is not None:
            x = self.ctx_att_block(x, ctx)
        x = self.mlp(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x = x.view(-1, L, C) # [B*V, L, C]
        cls_token = x[:, -1:] # [B*V, 1, C]
        cls_token = self.cross_att(x, query=cls_token) # [B*V, 1, C]
        cls_token = cls_token.reshape(B, V, -1, C) # [B, V, 1, C]

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token # [B, V, 1, C]
        m = category_token.shape[2] # num_class
        cls_token = cls_token.expand(B, V, m, C)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token) # sum(dot_prod(kc, mc)) -> m, where k = m

        distance = distance.mean(dim=1) # [B, m]
        return distance


class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x_full, pred_len):
        x = self.proj_in(x_full) # [B, V, L, C]
        x = self.mlp(x) # [B, V, L, C]
        x = self.proj_out(x) # [B, V, L, patch_len]

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # [B*V, L, patch_len]
        x = x.permute(0, 2, 1) # [B*V, patch_len, L]
        x = torch.nn.functional.fold(x, output_size=(
            pred_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1)) # [B*V, patch_len, pred_len]
        x = x.squeeze(dim=-1) # [B, V, patch_len, pred_len]
        x = x.reshape(bs, n_vars, -1) # [B, V, patch_len*pred_len]
        x = x.permute(0, 2, 1) # [B, patch_len*pred_len, V]
        return x


def calculate_reconstruct_loss(x_enc, mask_dec_out, mask):
    """
    x_enc: Original input tensor [B, L, V]
    mask_dec_out: Model's predictions [B, L, V]
    mask_seq: Mask tensor indicating which positions were masked [B, L] (1 for masked positions, 0 otherwise)
    """
    loss = (mask_dec_out - x_enc) ** 2 # [B, L, V]
    loss = loss.mean(dim=-1) # [B, L]
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss


class UniTS(nn.Module):
    """
    UniTS: Building a Unified Time Series Model
    d_model=256, stride=8, 
    patch_len=8, dropout=0.1, 
    e_layers=3, n_heads=8,
    min_mask_ratio=0.7, max_mask_ratio=0.8
    """

    def __init__(self, 
                 enc_in, num_class,
                 args,
                 task='cls',
                ):
        super().__init__()

        self.task = task
        if task == 'recon':
            self.min_mask_ratio = args.min_mask_ratio
            self.max_mask_ratio = args.max_mask_ratio
            self.phase = 'all'
        elif task == 'cls':
            self.phase = args.phase
        else:
            raise ValueError(f"Invalid task: {task}, should be 'recon', or 'cls'")

        # Tokens settings
        self.mask_token = nn.Parameter(torch.zeros(1, enc_in, 1, args.d_model))
        self.clip_token = nn.Parameter(torch.zeros(1, enc_in, 1, args.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, enc_in, 1, args.d_model))
        self.category_tokens = nn.Parameter(torch.zeros(1, enc_in, num_class, args.d_model))

        nn.init.normal_(self.clip_token, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.category_tokens, std=0.02)

        ### model settings ###
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)

        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        self.forecast_head = ForecastHead(args.d_model, args.patch_len, args.stride, args.stride, head_dropout=args.dropout)

        # Prior knowledge injection
        d_clip = 512
        self.ctx_proj = nn.Linear(d_clip, args.d_model)
        
        if self.task == 'cls':
            if self.phase == 'cls':
                for name, param in self.named_parameters():
                    if 'cls_head' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif self.phase == 'ft':
                for name, param in self.named_parameters():
                    if 'ctx' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def tokenize(self, x, mask=None):
        x = x.permute(0, 2, 1) # [B, V, L]
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, n_vars, padding

    def backbone(self, x, prior_emb=None):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prior_emb)
        return x
    
    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample random masking.
        """
        N, V, L, D = x.shape  # batch, var, length, dim

        # Calculate mask ratios and lengths to keep for each sample in the batch
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)

        # Create a range tensor and compare with len_keeps for mask generation
        range_tensor = torch.arange(L, device=x.device).expand(N, L)
        mask = (range_tensor >= len_keeps.unsqueeze(1))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.float()

        return mask # [B, L]
    
    def get_mask_seq(self, mask, seq_len):
        mask_seq = mask.unsqueeze(dim=-1).repeat(1, 1, self.patch_len)
        mask_seq = mask_seq.permute(0, 2, 1)
        mask_seq = mask_seq.masked_fill(mask_seq == 0, -1e9)
        # Fold operation
        mask_seq = torch.nn.functional.fold(mask_seq, output_size=(
            seq_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        # Apply threshold to bring back to 0/1 values
        mask_seq = (mask_seq > 0).float()
        mask_seq = mask_seq.squeeze(dim=-1).squeeze(dim=1)
        return mask_seq
    
    def classify(self, x, prior_emb=None):
        x_cls_prompt = self.cls_token.repeat(x.shape[0], 1, 1, 1)
        category_token = self.category_tokens.repeat(x.shape[0], 1, 1, 1)

        x = torch.cat((x, x_cls_prompt), dim=2)
        x = self.backbone(x, prior_emb) # [B, V, L, C]

        out = self.cls_head(x, category_token)

        return out
    
    def reconstruct(self, x, seqs, prior_emb=None, is_mask=True, lambda_recon=1.0):
        """
        aim to reconstruct the input x with context prior
        """
        # x: [B, V, L, C]
        mask_token = self.mask_token

        seq, seq_len, padding = seqs

        # mask
        if is_mask:
            mask = self.random_masking(x, self.min_mask_ratio, self.max_mask_ratio) # [B, L']
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1) # [B, 1, L', 1]
            mask_repeat = mask_repeat.repeat(1, x.shape[1], 1, x.shape[-1]) # [B, V, L', C]
            x = x * (1-mask_repeat) + mask_token * mask_repeat # removed token is 0

            mask_seq = self.get_mask_seq(mask, seq_len+padding)
            mask_seq = mask_seq[:, :seq_len]
        else:
            mask_seq = None
        
        x = self.backbone(x, prior_emb) # [B, V, L, C]

        mask_dec_out = self.forecast_head(x, seq_len+padding)
        mask_dec_out = mask_dec_out[:, :seq_len] # [B, L, V]

        r_loss = lambda_recon * calculate_reconstruct_loss(seq, mask_dec_out, mask_seq)
        return r_loss
    
    def forward(self, x, prior_emb=None):
        # x: [B, L, V]
        seq_len = x.shape[1]
        seq_src = x.detach().clone()
        x, n_vars, pad = self.tokenize(x) # [B*V, L, C]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # [B, V, L, C]
        x = x + self.position_embedding(x)

        if prior_emb is not None:
            prior_emb = self.ctx_proj(prior_emb)
        
        if self.task == 'cls':
            return self.classify(x, prior_emb)
        elif self.task == 'recon':
            seqs = (seq_src, seq_len, pad)
            return self.reconstruct(x, seqs, prior_emb)

@dataclass
class UniTSArgs:
    d_model: int
    n_heads: int
    e_layers: int
    patch_len: int
    stride: int
    dropout: float
    phase: str
    load_path: str = None
    # pretrain
    min_mask_ratio: float = None
    max_mask_ratio: float = None

    def __post_init__(self):
        self.phase = self.phase.lower()
    
    @classmethod
    def from_args(cls, args):
        # ignore args that are not in the dataclass
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_dict(cls, args):
        return cls(**args)
    
    def to_dict(self):
        return vars(self)
