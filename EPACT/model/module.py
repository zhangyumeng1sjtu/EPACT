import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SelfAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, key_dim=None, value_dim=None, dropout=0., bias=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.key_dim = key_dim if key_dim is not None else embed_dim
        self.value_dim = value_dim if value_dim is not None else embed_dim
        self.qkv_same_dim = self.key_dim == embed_dim and self.value_dim == embed_dim
        assert self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size")

        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, (
                "embed_dim must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.key_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.value_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                need_head_weights: bool = False
                ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, batch_size, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scale

        q = q.contiguous().view(tgt_len, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.num_heads,
                                self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == batch_size
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [
            batch_size * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(
                    2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [batch_size *
                                     self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(
            tgt_len, batch_size, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(
                batch_size, self.num_heads, tgt_len, src_len
            ).type_as(attn)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights


class TransformerLayer(nn.Module):

    def __init__(self, embed_dim, ffn_embed_dim, num_heads, add_bias_kv=False, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout

        # self.self_attn = SelfAttentionLayer(self.embed_dim, self.num_heads, dropout=attn_dropout)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.attn_layernorm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.ffn_embed_dim),
            nn.GELU(),
            nn.Linear(self.ffn_embed_dim, self.embed_dim),
        )

    def forward(self, x, attn_mask=None, padding_mask=None):

        residual = x
        x = self.attn_layernorm(x)  # pre-LN
        x, attn = self.self_attn(
            x, x, x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=True
        )
        x = residual + x

        residual = x
        x = self.ffn(x)
        x = residual + x

        return x, attn


class LMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(
            mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        
class LearnedChainPositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, selected_chain_idx: int):
        super(LearnedChainPositionalEmbedding, self).__init__()
        self.max_positions = num_embeddings
        self.embedding_dim = embedding_dim
        self.selected_chain_idx = selected_chain_idx
        
        self.embedding_layer = nn.Embedding(self.max_positions, self.embedding_dim)

    def forward(self, chain_mask: torch.Tensor):
        mask = chain_mask.eq(self.selected_chain_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()
        
        return self.embedding_layer(positions)


class ResidueConvBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size,
                      stride=1, padding=padding),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x = residual + x

        return x


class MHCConvLayers(nn.Module):
    
    def __init__(self, num_layers, in_dim, embed_dim, mhc_len, kernel_size):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.mhc_len = mhc_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)
        
    def forward(self, x):  # x: B, L, E
        x = x.transpose(1, 2)  # B, L, E => B, E, L
        x = self.conv(x)
        x = self.bn(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return x.transpose(1, 2) # B, E, L => B, L, E


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.attn_layer = nn.Sequential(
            # nn.Linear(embed_dim, embed_dim),
            # nn.Tanh(),
            # only use a single linear layer
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, x, mask=None):
        attn = self.attn_layer(x.transpose(1,2)) # B, E, L => B, L, 1
        attn = attn.masked_fill(mask.unsqueeze(2).to(torch.bool), float("-inf"))
        attn = F.softmax(attn, dim=1)
        pooled = torch.bmm(x, attn).squeeze(2) # B, E, L x B, L, 1 => B, E
        return pooled, attn


class DistPredictor(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.dist_map_predictor = nn.Conv2d(
            in_channels=hid_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, tcr_feat, epi_feat, tcr_padding_mask, epi_padding_mask):
        inter_padding_mask = torch.logical_or(tcr_padding_mask[:, 1:].unsqueeze(2), epi_padding_mask[:, 1:].unsqueeze(1))  # B, L1-1, L2-1
        tcr_feat_mat = tcr_feat[:, :, 1:].unsqueeze(3).repeat([1, 1, 1, epi_feat.shape[2]-1]) # B, E, L1-1, L2-1
        epi_feat_mat = epi_feat[:, :, 1:].unsqueeze(2).repeat([1, 1, tcr_feat.shape[2]-1, 1]) # B, E, L1-1, L2-1
        
        inter_map = tcr_feat_mat * epi_feat_mat  # B, E, L1-1, L2-1
        
        inter_map = inter_map.masked_fill(inter_padding_mask.unsqueeze(1), float("-inf"))
        dist_map = self.dist_map_predictor(F.relu(inter_map))
        out_dist = F.relu(dist_map[:, 0, :, :])
        out_contact = F.sigmoid(dist_map[:, 1, :, :])
        
        return torch.cat([out_dist.unsqueeze(-1), out_contact.unsqueeze(-1)], dim=-1)
        