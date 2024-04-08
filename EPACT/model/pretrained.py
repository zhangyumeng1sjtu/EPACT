from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import LMHead, LearnedPositionalEmbedding, LearnedChainPositionalEmbedding, MHCConvLayers, TransformerLayer
from ..utils.tokenizer import IUPACTokenizer


class ProteinLM(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 4,
        max_seq_length: int = 26,
        attn_dropout: float = 0.0,
        token_dropout: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.token_dropout = token_dropout
        self.attn_dropout = attn_dropout
        
        self.tokenizer = IUPACTokenizer()
            
        self.alphabet_size = len(self.tokenizer)
        self.padding_idx = self.tokenizer.padding_idx
        self.mask_idx = self.tokenizer.mask_idx
        self.cls_idx = self.tokenizer.cls_idx
        self.eos_idx = self.tokenizer.eos_idx
        
        self._init_modules()
    
    @abstractmethod
    def _init_position_encoding(self):
        raise NotImplementedError
    
    @abstractmethod
    def _forward_position_encoding(self, tokens, chain_token_mask):
        raise NotImplementedError
    
    def _init_modules(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        
        self._init_position_encoding()
        
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    2 * self.embed_dim,
                    self.num_heads,
                    add_bias_kv=False,
                    attn_dropout=self.attn_dropout
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.lm_head = LMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        
        self.atchely_factor_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 5)
        )
        
    def forward(self, tokens, chain_token_mask=None):
        padding_mask = tokens.eq(self.padding_idx)  # B, L
        x = self.embed_tokens(tokens)  # B, L, E

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (
                tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / \
                (1 - mask_ratio_observed)[:, None, None]

        x += self._forward_position_encoding(tokens, chain_token_mask)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)  # B, L, E => L, B, E

        if not padding_mask.any():
            padding_mask = None

        for layer in self.layers:
            x, attn = layer(x, padding_mask=padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # L, B, E => B, L, E

        logits = self.lm_head(x)
        atchely_factor = self.atchely_factor_head(x)
        
        result = {'logits': logits, 'atchely_factor': atchely_factor, 'rep': x}

        return result
  
        
class PeptideLM(ProteinLM):
    
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 4,
        max_seq_length: int = 26,
        attn_dropout: float = 0.0,
        token_dropout: bool = False,
    ):
        super(PeptideLM, self).__init__(num_layers, embed_dim, num_heads,
                                        max_seq_length, attn_dropout, token_dropout)
        
    def _init_position_encoding(self):
        self.embed_positions = LearnedPositionalEmbedding(
            self.max_seq_length, self.embed_dim, self.padding_idx
        )

    def _forward_position_encoding(self, tokens, chain_token_mask):
        return self.embed_positions(tokens)

 
class TCRCDR3LM(ProteinLM):
    
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 4,
        max_seq_length: int = 26,
        attn_dropout: float = 0.0,
        token_dropout: bool = False,
    ):
        super(TCRCDR3LM, self).__init__(num_layers, embed_dim, num_heads,
                                        max_seq_length, attn_dropout, token_dropout)
    
    def _init_position_encoding(self):
        self.embed_positions_alpha = LearnedChainPositionalEmbedding(
            self.max_seq_length, self.embed_dim, selected_chain_idx=1
        )
        self.embed_positions_beta = LearnedChainPositionalEmbedding(
            self.max_seq_length, self.embed_dim, selected_chain_idx=2
        )

    def  _forward_position_encoding(self, tokens, chain_token_mask):
        return self.embed_positions_alpha(chain_token_mask) + self.embed_positions_beta(chain_token_mask)


class TCRCDR123LM(ProteinLM):
    
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 4,
        max_seq_length: int = 26,
        attn_dropout: float = 0.0,
        token_dropout: bool = False
    ):
        super(TCRCDR123LM, self).__init__(num_layers, embed_dim, num_heads,
                                        max_seq_length, attn_dropout, token_dropout)
    
    def _init_position_encoding(self):
        self.embed_positions_cdr1a = LearnedChainPositionalEmbedding(
            8, self.embed_dim, selected_chain_idx=1
        )
        self.embed_positions_cdr2a = LearnedChainPositionalEmbedding(
            9, self.embed_dim, selected_chain_idx=2
        )
        self.embed_positions_cdr3a = LearnedChainPositionalEmbedding(
            self.max_seq_length, self.embed_dim, selected_chain_idx=3
        )
        self.embed_positions_cdr1b = LearnedChainPositionalEmbedding(
            7, self.embed_dim, selected_chain_idx=4
        )
        self.embed_positions_cdr2b = LearnedChainPositionalEmbedding(
            8, self.embed_dim, selected_chain_idx=5
        )
        self.embed_positions_cdr3b = LearnedChainPositionalEmbedding(
            self.max_seq_length, self.embed_dim, selected_chain_idx=6
        )
        
    def _forward_position_encoding(self, tokens, chain_token_mask):
        return self.embed_positions_cdr1a(chain_token_mask) + self.embed_positions_cdr2a(chain_token_mask) + self.embed_positions_cdr3a(chain_token_mask) + \
                self.embed_positions_cdr1b(chain_token_mask) + self.embed_positions_cdr2b(chain_token_mask) + self.embed_positions_cdr3b(chain_token_mask)


class EpitopeMHCModel(nn.Module):
    
    def __init__(
        self,
        num_epi_layers: int = 6,
        num_epi_heads: int = 4,
        embed_epi_dim: int = 512,
        num_mhc_layers: int = 4,
        in_mhc_dim: int = 45,
        embed_mhc_dim: int = 256,
        cross_attn_heads: int = 4,
        mhc_seq_len: int = 34,
        attn_dropout: float = 0.05,
        dropout: float = 0.
    ):
        super().__init__()

        self.epitope_model = PeptideLM(
            num_layers=num_epi_layers,
            embed_dim=embed_epi_dim,
            num_heads=num_epi_heads,
            attn_dropout=attn_dropout
        )
        self.mhc_model = MHCConvLayers( # add dropout at each layer
            num_layers=num_mhc_layers,
            in_dim=in_mhc_dim,
            embed_dim=embed_mhc_dim,
            mhc_len=mhc_seq_len,
            kernel_size=3
        )
        self.cross_attn_epi_mhc = nn.MultiheadAttention(
            embed_dim=embed_epi_dim,
            num_heads=cross_attn_heads,
            dropout=attn_dropout,
            kdim=embed_mhc_dim,
            vdim=embed_mhc_dim,
            batch_first=True
        )
        self.cross_attn_mhc_epi = nn.MultiheadAttention(
            embed_dim=embed_mhc_dim,
            num_heads=cross_attn_heads,
            dropout=attn_dropout,
            kdim=embed_epi_dim,
            vdim=embed_epi_dim,
            batch_first=True
        )
        self.epi_layer_norm =  nn.LayerNorm(embed_epi_dim)
        self.mhc_layer_norm =  nn.LayerNorm(embed_mhc_dim)
        self.lm_head = LMHead(
            embed_dim=embed_epi_dim,
            output_dim=self.epitope_model.alphabet_size,
            weight=self.epitope_model.embed_tokens.weight
        )
        self.mhc_attn = nn.Sequential(
            nn.Linear(embed_mhc_dim, embed_mhc_dim),
            nn.Tanh(),
            nn.Linear(embed_mhc_dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_epi_dim + embed_mhc_dim, embed_epi_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_epi_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def load_pretrained_epitope_model(self, pretrained_model_path):
        # Load the pre-trained epitope LM model and freeze weights
        self.epitope_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        lm_head_state_dict = self.epitope_model.lm_head.state_dict()
        self.lm_head.load_state_dict(lm_head_state_dict)
        
        for param in self.epitope_model.parameters():
            param.requires_grad = False
    
    def forward(self, epi, mhc, return_epi_embedding=False):
        
        epi_feat = self.epitope_model(epi)['rep']  # B, Lp, Ep
        
        padding_mask = epi.eq(self.epitope_model.padding_idx) # B, Lp
        if not padding_mask.any():
            padding_mask = None
        mhc_feat = self.mhc_model(mhc)  # B, Lm, Em
        
        epi_feat_, attn_epi_mhc = self.cross_attn_epi_mhc(query=epi_feat.clone(), key=mhc_feat.clone(), value=mhc_feat.clone(), key_padding_mask=None)
        mhc_feat_, attn_mhc_epi = self.cross_attn_mhc_epi(query=mhc_feat.clone(), key=epi_feat.clone(), value=epi_feat.clone(), key_padding_mask=padding_mask)
    
        epi_feat += self.epi_layer_norm(epi_feat_)
        mhc_feat += self.mhc_layer_norm(mhc_feat_)
        
        epi_feat += epi_feat_
        mhc_feat += mhc_feat_
        
        if return_epi_embedding:
            return epi_feat
        
        epi_logits = self.lm_head(epi_feat)
        
        mhc_attn = F.softmax(self.mhc_attn(mhc_feat), dim=1) # B, Lm, Em => B, Lm, 1
        mhc_pooled = torch.bmm(mhc_feat.transpose(1,2), mhc_attn).squeeze(2) # B, Em, Lm x B, Lm, 1 => B, Em
        
        epi_cls = epi_feat[:, 0, :].squeeze(1) # B, Lp, Ep => B, Ep
        pmhc_embed = torch.cat([epi_cls, mhc_pooled], dim=1) # B, Ep+Em
        out = self.mlp(pmhc_embed)
        
        return epi_logits, out
      