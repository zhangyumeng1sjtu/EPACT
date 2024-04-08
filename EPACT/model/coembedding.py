import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import AttentionPooling, DistPredictor, ResidueConvBlock
from ..utils.tokenizer import IUPACTokenizer
from .pretrained import EpitopeMHCModel, TCRCDR3LM, TCRCDR123LM


class PairedCDR3pMHCCoembeddingModel(nn.Module):
    def __init__(
        self,
        num_epi_layers: int = 6,
        num_epi_heads: int = 4,
        embed_epi_dim: int = 512,
        num_mhc_layers: int = 4,
        in_mhc_dim: int = 45,
        embed_mhc_dim: int = 256,
        mhc_seq_len: int = 34,
        cross_attn_heads: int = 4, 
        num_tcr_layers: int = 6,
        num_tcr_heads: int = 4,
        embed_tcr_dim: int = 512,
        num_attn_heads: int = 4,
        embed_hid_dim: int = 512,
        num_conv_layers: int = 2,
        attn_dropout: float = 0.05,
        dropout: float = 0.,
        projector_type: str = 'linear', 
        agg: str = 'cls',
    ):
        super().__init__()
        
        self.tokenizer = IUPACTokenizer()
        self.tcr_padding_idx = self.tokenizer.padding_idx
        self.tcr_sep_idx = self.tokenizer.sep_idx
        self.epitope_padding_idx = self.tokenizer.padding_idx
        
        self.tcr_model = TCRCDR3LM(num_tcr_layers, embed_tcr_dim, num_tcr_heads, max_seq_length=26, attn_dropout=attn_dropout)
        self.pmhc_model = EpitopeMHCModel(
            num_epi_layers=num_epi_layers,
            num_epi_heads=num_epi_heads,
            embed_epi_dim=embed_epi_dim,
            num_mhc_layers=num_mhc_layers,
            in_mhc_dim=in_mhc_dim,
            embed_mhc_dim=embed_mhc_dim,
            cross_attn_heads=cross_attn_heads,
            mhc_seq_len=mhc_seq_len,
            attn_dropout=attn_dropout
        ) 
        
        self.tcr_conv = nn.Conv1d(embed_tcr_dim, embed_hid_dim, 1, 1, bias=False)
        self.epi_conv = nn.Conv1d(embed_epi_dim, embed_hid_dim, 1, 1, bias=False)
        
        self.epi_self_attn = nn.MultiheadAttention(
            embed_dim=embed_hid_dim,
            num_heads=num_attn_heads,
            dropout=attn_dropout, 
            batch_first=True
        )
        self.tcr_self_attn = nn.MultiheadAttention(
            embed_dim=embed_hid_dim,
            num_heads=num_attn_heads,
            dropout=attn_dropout, 
            batch_first=True
        )
        self.epi_layer_norm =  nn.LayerNorm(embed_hid_dim)
        self.tcr_layer_norm =  nn.LayerNorm(embed_hid_dim)
        
        self.epi_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=embed_hid_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_conv_layers)
            ]
        )
        self.tcr_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=embed_hid_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_conv_layers)
            ]
        )
        
        self.projector_type = projector_type
        if projector_type == 'linear':
            self.tcr_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.ReLU())
            nn.init.xavier_normal_(self.tcr_projector[0].weight)
            self.epi_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.ReLU())
            nn.init.xavier_normal_(self.epi_projector[0].weight)
        elif projector_type == 'mlp':
            self.tcr_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.BatchNorm1d(embed_hid_dim), nn.ReLU(), nn.Linear(embed_hid_dim, embed_hid_dim))
            self.epi_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.BatchNorm1d(embed_hid_dim), nn.ReLU(), nn.Linear(embed_hid_dim, embed_hid_dim))
        
        self.agg = agg
        if self.agg == 'attn':
            self.tcr_attn_pool = AttentionPooling(embed_dim=embed_hid_dim)
            self.epi_attn_pool = AttentionPooling(embed_dim=embed_hid_dim)
            
        self.clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_hid_dim, 1),
            nn.Sigmoid()
        )
        
        self.dist_predictor = DistPredictor(hid_dim=embed_hid_dim, out_dim=2)
        
    def load_pretrained_pmhc_model(self, pretrained_model_path):
        self.pmhc_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        for param in self.pmhc_model.epitope_model.parameters():
            param.requires_grad = False  
        for param in self.pmhc_model.mhc_model.parameters():
            param.requires_grad = False
        # for param in self.pmhc_model.parameters():
        #     param.requires_grad = False
            
    def load_pretrained_tcr_model(self, pretrained_model_path):
        self.tcr_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        for param in self.tcr_model.parameters():
            param.requires_grad = False
    
    def compute_pmhc_feat(self, pmhc_data):
        epi = pmhc_data['epitope_token']
        mhc = pmhc_data['mhc_embedding']
        epi_feat = self.pmhc_model(epi, mhc, return_epi_embedding=True)  
        epi_padding_mask = epi.eq(self.epitope_padding_idx)
        return epi_feat, epi_padding_mask
    
    def compute_pmhc_embedding(self, epi_feat, epi_padding_mask=None):
        epi_feat = self.epi_conv(epi_feat.transpose(1, 2)).transpose(1, 2) # B, L2, E
        epi_feat_, attn_epi = self.epi_self_attn(epi_feat.clone(), epi_feat.clone(), epi_feat.clone(), key_padding_mask=epi_padding_mask)
        epi_feat += self.epi_layer_norm(epi_feat_) # B, L2, E
        epi_feat = epi_feat.transpose(1,2) # B, E, L2
        for layer in self.epi_conv_layers:
            epi_feat = layer(epi_feat)
            
        if self.agg == 'attn':
            epi_projection, epi_attn = self.epi_attn_pool(epi_feat[:, :, 1:], mask=epi_padding_mask[:, 1:])
        elif self.agg == 'mean':
            sum_epi_pooled = torch.sum(epi_feat[:, :, 1:] * (~epi_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            epi_projection = sum_epi_pooled / torch.sum(~epi_padding_mask[:, 1:], dim=1, keepdim=True)
        else:
            epi_projection = epi_feat[:, :, 0].squeeze(1) 
            
        return epi_projection, self.epi_projector(epi_projection)
        
    def compute_tcr_feat(self, tcr, chain_mask=None):
        tcr_feat = self.tcr_model(tcr, chain_mask)['rep']
        tcr_padding_mask = tcr.eq(self.tcr_padding_idx)
        tcr_sep_mask = tcr.eq(self.tcr_sep_idx) # add sep mask
        return tcr_feat, tcr_padding_mask, tcr_sep_mask
    
    def compute_tcr_embedding(self, tcr_feat, tcr_padding_mask=None):
        tcr_feat = self.tcr_conv(tcr_feat.transpose(1, 2)).transpose(1, 2) # B, L1, E
        tcr_feat_, attn_tcr = self.tcr_self_attn(tcr_feat.clone(), tcr_feat.clone(), tcr_feat.clone(), key_padding_mask=tcr_padding_mask)
        tcr_feat += self.tcr_layer_norm(tcr_feat_) # B, L1, E
        tcr_feat = tcr_feat.transpose(1,2) # B, E, L1
        for layer in self.tcr_conv_layers:
            tcr_feat = layer(tcr_feat)
            
        if self.agg == 'attn':
            tcr_projection, tcr_attn = self.tcr_attn_pool(tcr_feat[:, :, 1:], mask=tcr_padding_mask[:, 1:])
        elif self.agg == 'mean':
            sum_tcr_pooled = torch.sum(tcr_feat[:, :, 1:] * (~tcr_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            tcr_projection = sum_tcr_pooled / torch.sum(~tcr_padding_mask[:, 1:], dim=1, keepdim=True)
        else:
            tcr_projection = tcr_feat[:, :, 0].squeeze(1) 
            
        return tcr_projection, self.tcr_projector(tcr_projection)
    
    def get_logits(self, tcr_embed, epi_embed):
        embed = torch.cat([tcr_embed, epi_embed], dim=1) 
        logits = self.clf(embed)
        return logits
    
    def forward(self, epi_feat, tcr_feat,
                epi_padding_mask=None, tcr_padding_mask=None, tcr_sep_mask=None, predict_dist_map=False):
        
        tcr_feat = self.tcr_conv(tcr_feat.transpose(1, 2)).transpose(1, 2) # B, L1, E
        epi_feat = self.epi_conv(epi_feat.transpose(1, 2)).transpose(1, 2) # B, L2, E
        
        tcr_feat_, attn_tcr = self.tcr_self_attn(tcr_feat.clone(), tcr_feat.clone(), tcr_feat.clone(), key_padding_mask=tcr_padding_mask)
        epi_feat_, attn_epi = self.epi_self_attn(epi_feat.clone(), epi_feat.clone(), epi_feat.clone(), key_padding_mask=epi_padding_mask)
        
        tcr_feat += self.tcr_layer_norm(tcr_feat_) # B, L1, E
        epi_feat += self.epi_layer_norm(epi_feat_) # B, L2, E
            
        tcr_feat = tcr_feat.transpose(1,2) # B, E, L1
        epi_feat = epi_feat.transpose(1,2) # B, E, L2
        
        for layer in self.tcr_conv_layers:
            tcr_feat = layer(tcr_feat)
        for layer in self.epi_conv_layers:
            epi_feat = layer(epi_feat)
        
        if predict_dist_map:            
            if tcr_sep_mask is not None:
                tcr_mask = torch.logical_or(tcr_padding_mask, tcr_sep_mask)
            return self.dist_predictor(tcr_feat, epi_feat, tcr_mask, epi_padding_mask)
        
        # attention pooling
        if self.agg == 'attn':
            tcr_projection, tcr_attn = self.tcr_attn_pool(tcr_feat[:, :, 1:], mask=tcr_padding_mask[:, 1:])
            epi_projection, epi_attn = self.epi_attn_pool(epi_feat[:, :, 1:], mask=epi_padding_mask[:, 1:])    
        # mean pooling
        elif self.agg == 'mean':
            sum_tcr_pooled = torch.sum(tcr_feat[:, :, 1:] * (~tcr_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            tcr_projection = sum_tcr_pooled / torch.sum(~tcr_padding_mask[:, 1:], dim=1, keepdim=True)
            sum_epi_pooled = torch.sum(epi_feat[:, :, 1:] * (~epi_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            epi_projection = sum_epi_pooled / torch.sum(~epi_padding_mask[:, 1:], dim=1, keepdim=True)    
        else:
            tcr_projection = tcr_feat[:, :, 0].squeeze(1) # B, L1, E => B, E
            epi_projection = epi_feat[:, :, 0].squeeze(1) # B, L2, E => B, E
        
        embed = torch.cat([tcr_projection, epi_projection], dim=1) 
        logits = self.clf(embed)
        
        tcr_projection = self.tcr_projector(tcr_projection)
        epi_projection = self.epi_projector(epi_projection)
        dist = F.cosine_similarity(tcr_projection, epi_projection, dim=-1)
    
        output = {'logits': logits, 'dist': dist, 'projection': [tcr_projection, epi_projection]}
        if self.agg == 'attn':
            output['attn'] = [tcr_attn, epi_attn]
            
        return output


class PairedCDR123pMHCCoembeddingModel(nn.Module):
    def __init__(
        self,
        num_epi_layers: int = 6,
        num_epi_heads: int = 4,
        embed_epi_dim: int = 512,
        num_mhc_layers: int = 4,
        in_mhc_dim: int = 45,
        embed_mhc_dim: int = 256,
        mhc_seq_len: int = 34,
        cross_attn_heads: int = 4, 
        num_tcr_layers: int = 6,
        num_tcr_heads: int = 4,
        embed_tcr_dim: int = 512,
        num_attn_heads: int = 4,
        embed_hid_dim: int = 512,
        num_conv_layers: int = 2,
        attn_dropout: float = 0.05,
        dropout: float = 0.,
        projector_type: str = 'linear', 
        agg: str = 'cls',
    ):
        super().__init__()
        
        self.tokenizer = IUPACTokenizer()
        self.tcr_padding_idx = self.tokenizer.padding_idx
        self.tcr_sep_idx = self.tokenizer.sep_idx
        self.epitope_padding_idx = self.tokenizer.padding_idx

        self.tcr_model = TCRCDR123LM(num_tcr_layers, embed_tcr_dim, num_tcr_heads, max_seq_length=26, attn_dropout=attn_dropout)
     
        self.pmhc_model = EpitopeMHCModel(
            num_epi_layers=num_epi_layers,
            num_epi_heads=num_epi_heads,
            embed_epi_dim=embed_epi_dim,
            num_mhc_layers=num_mhc_layers,
            in_mhc_dim=in_mhc_dim,
            embed_mhc_dim=embed_mhc_dim,
            cross_attn_heads=cross_attn_heads,
            mhc_seq_len=mhc_seq_len,
            attn_dropout=attn_dropout
        ) 
        
        self.tcr_conv = nn.Conv1d(embed_tcr_dim, embed_hid_dim, 1, 1, bias=False)
        self.epi_conv = nn.Conv1d(embed_epi_dim, embed_hid_dim, 1, 1, bias=False)
        
        self.epi_self_attn = nn.MultiheadAttention(
            embed_dim=embed_hid_dim,
            num_heads=num_attn_heads,
            dropout=attn_dropout, 
            batch_first=True
        )
        self.tcr_self_attn = nn.MultiheadAttention(
            embed_dim=embed_hid_dim,
            num_heads=num_attn_heads,
            dropout=attn_dropout, 
            batch_first=True
        )
        self.epi_layer_norm =  nn.LayerNorm(embed_hid_dim)
        self.tcr_layer_norm =  nn.LayerNorm(embed_hid_dim)
        
        self.epi_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=embed_hid_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_conv_layers)
            ]
        )
        self.tcr_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=embed_hid_dim, kernel_size=1, padding=0, dropout=dropout) # add dropout
                for _ in range(num_conv_layers)
            ]
        )
        
        self.projector_type = projector_type
        if projector_type == 'linear':
            self.tcr_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.ReLU())
            nn.init.xavier_normal_(self.tcr_projector[0].weight)
            self.epi_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.ReLU())
            nn.init.xavier_normal_(self.epi_projector[0].weight)
        elif projector_type == 'mlp':
            self.tcr_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.BatchNorm1d(embed_hid_dim), nn.ReLU(), nn.Linear(embed_hid_dim, embed_hid_dim))
            self.epi_projector = nn.Sequential(nn.Linear(embed_hid_dim, embed_hid_dim), nn.BatchNorm1d(embed_hid_dim), nn.ReLU(), nn.Linear(embed_hid_dim, embed_hid_dim))
        
        self.agg = agg
        if self.agg == 'attn':
            self.tcr_attn_pool = AttentionPooling(embed_dim=embed_hid_dim)
            self.epi_attn_pool = AttentionPooling(embed_dim=embed_hid_dim)
            
        self.clf = nn.Sequential(
            nn.Linear(embed_hid_dim * 2, embed_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_hid_dim, 1),
            nn.Sigmoid()
        )
        
        self.dist_predictor = DistPredictor(hid_dim=embed_hid_dim, out_dim=2)
        self.selected_segments = [1, 3, 6]
        
    def load_pretrained_pmhc_model(self, pretrained_model_path):
        self.pmhc_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        for param in self.pmhc_model.epitope_model.parameters():
            param.requires_grad = False  
        for param in self.pmhc_model.mhc_model.parameters():
            param.requires_grad = False
        # for param in self.pmhc_model.parameters():
        #     param.requires_grad = False
            
    def load_pretrained_tcr_model(self, pretrained_model_path):
        self.tcr_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
        for param in self.tcr_model.parameters():
            param.requires_grad = False
    
    def compute_pmhc_feat(self, pmhc_data):
        epi = pmhc_data['epitope_token']
        mhc = pmhc_data['mhc_embedding']
        epi_feat = self.pmhc_model(epi, mhc, return_epi_embedding=True)  
        epi_padding_mask = epi.eq(self.epitope_padding_idx)
        return epi_feat, epi_padding_mask
    
    def compute_pmhc_embedding(self, epi_feat, epi_padding_mask=None):
        epi_feat = self.epi_conv(epi_feat.transpose(1, 2)).transpose(1, 2) # B, L2, E
        epi_feat_, attn_epi = self.epi_self_attn(epi_feat.clone(), epi_feat.clone(), epi_feat.clone(), key_padding_mask=epi_padding_mask)
        epi_feat += self.epi_layer_norm(epi_feat_) # B, L2, E
        epi_feat = epi_feat.transpose(1,2) # B, E, L2
        for layer in self.epi_conv_layers:
            epi_feat = layer(epi_feat)
            
        if self.agg == 'attn':
            epi_projection, epi_attn = self.epi_attn_pool(epi_feat[:, :, 1:], mask=epi_padding_mask[:, 1:])
        elif self.agg == 'mean':
            sum_epi_pooled = torch.sum(epi_feat[:, :, 1:] * (~epi_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            epi_projection = sum_epi_pooled / torch.sum(~epi_padding_mask[:, 1:], dim=1, keepdim=True)
        else:
            epi_projection = epi_feat[:, :, 0].squeeze(1) 
            
        return epi_projection, self.epi_projector(epi_projection)
        
    def compute_tcr_feat(self, tcr, segment_mask=None):
        tcr_feat = self.tcr_model(tcr, segment_mask)['rep']
        tcr_padding_mask = tcr.eq(self.tcr_padding_idx)
        return tcr_feat, tcr_padding_mask
    
    def compute_tcr_embedding(self, tcr_feat, tcr_padding_mask=None):
        tcr_feat = self.tcr_conv(tcr_feat.transpose(1, 2)).transpose(1, 2) # B, L1, E
        tcr_feat_, attn_tcr = self.tcr_self_attn(tcr_feat.clone(), tcr_feat.clone(), tcr_feat.clone(), key_padding_mask=tcr_padding_mask)
        tcr_feat += self.tcr_layer_norm(tcr_feat_) # B, L1, E
        tcr_feat = tcr_feat.transpose(1,2) # B, E, L1
        for layer in self.tcr_conv_layers:
            tcr_feat = layer(tcr_feat)
            
        if self.agg == 'attn':
            tcr_projection, tcr_attn = self.tcr_attn_pool(tcr_feat[:, :, 1:], mask=tcr_padding_mask[:, 1:])
        elif self.agg == 'mean':
            sum_tcr_pooled = torch.sum(tcr_feat[:, :, 1:] * (~tcr_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            tcr_projection = sum_tcr_pooled / torch.sum(~tcr_padding_mask[:, 1:], dim=1, keepdim=True)
        else:
            tcr_projection = tcr_feat[:, :, 0].squeeze(1) 
            
        return tcr_projection, self.tcr_projector(tcr_projection)
    
    def get_logits(self, tcr_embed, epi_embed):
        embed = torch.cat([tcr_embed, epi_embed], dim=1) 
        logits = self.clf(embed)
        return logits
    
    def forward(self, epi_feat, tcr_feat,
                epi_padding_mask=None, tcr_padding_mask=None, predict_dist_map=False, tcr_segment_mask=None):
        
        tcr_feat = self.tcr_conv(tcr_feat.transpose(1, 2)).transpose(1, 2) # B, L1, E
        epi_feat = self.epi_conv(epi_feat.transpose(1, 2)).transpose(1, 2) # B, L2, E
        
        tcr_feat_, attn_tcr = self.tcr_self_attn(tcr_feat.clone(), tcr_feat.clone(), tcr_feat.clone(), key_padding_mask=tcr_padding_mask)
        epi_feat_, attn_epi = self.epi_self_attn(epi_feat.clone(), epi_feat.clone(), epi_feat.clone(), key_padding_mask=epi_padding_mask)
        
        tcr_feat += self.tcr_layer_norm(tcr_feat_) # B, L1, E
        epi_feat += self.epi_layer_norm(epi_feat_) # B, L2, E
            
        tcr_feat = tcr_feat.transpose(1,2) # B, E, L1
        epi_feat = epi_feat.transpose(1,2) # B, E, L2
        
        for layer in self.tcr_conv_layers:
            tcr_feat = layer(tcr_feat)
        for layer in self.epi_conv_layers:
            epi_feat = layer(epi_feat)
        
        if predict_dist_map:
            # add padding mask to non-selected segments
            not_selected_segment_mask = ~tcr_segment_mask.unsqueeze(2).eq(torch.tensor(self.selected_segments, device=tcr_segment_mask.device)).any(dim=2)
            tcr_padding_mask = torch.logical_or(tcr_padding_mask, not_selected_segment_mask)
            return self.dist_predictor(tcr_feat, epi_feat, tcr_padding_mask, epi_padding_mask)
        
        # attention pooling
        if self.agg == 'attn':
            tcr_projection, tcr_attn = self.tcr_attn_pool(tcr_feat[:, :, 1:], mask=tcr_padding_mask[:, 1:])
            epi_projection, epi_attn = self.epi_attn_pool(epi_feat[:, :, 1:], mask=epi_padding_mask[:, 1:])    
        # mean pooling
        elif self.agg == 'mean':
            sum_tcr_pooled = torch.sum(tcr_feat[:, :, 1:] * (~tcr_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            tcr_projection = sum_tcr_pooled / torch.sum(~tcr_padding_mask[:, 1:], dim=1, keepdim=True)
            sum_epi_pooled = torch.sum(epi_feat[:, :, 1:] * (~epi_padding_mask[:, 1:].unsqueeze(1)), dim=-1)
            epi_projection = sum_epi_pooled / torch.sum(~epi_padding_mask[:, 1:], dim=1, keepdim=True)    
        else:
            tcr_projection = tcr_feat[:, :, 0].squeeze(1) # B, L1, E => B, E
            epi_projection = epi_feat[:, :, 0].squeeze(1) # B, L2, E => B, E
        
        embed = torch.cat([tcr_projection, epi_projection], dim=1) 
        logits = self.clf(embed)
        
        tcr_projection = self.tcr_projector(tcr_projection)
        epi_projection = self.epi_projector(epi_projection)
        dist = F.cosine_similarity(tcr_projection, epi_projection, dim=-1)
    
        output = {'logits': logits, 'dist': dist, 'projection': [tcr_projection, epi_projection]}
        if self.agg == 'attn':
            output['attn'] = [tcr_attn, epi_attn]
            
        return output
    