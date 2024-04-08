from typing import Sequence

import torch
import numpy as np

from ..utils.sampling import sample_binding_cdr3, sample_non_binding_cdr3
from ..utils.misc import load_paired_tcr_feat_bank, load_paired_cdr123_feat_bank
from ..utils.tokenizer import IUPACTokenizer
from ..utils.encoding import encoding_dist, encoding_paired_dist, epitope_batch_encoding, paired_cdr3_batch_encoding, mhc_encoding, paired_cdr123_batch_encoding


class EpitopeBatchConverter(object):

    def __init__(self, max_epitope_len=15, use_atchley_factor=False, mask_prob=0.15):  # 15 for epitopes binding with MHC class I
        self.tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.use_atchley_factor = use_atchley_factor 
        self.mask_prob = mask_prob

    def __call__(self, raw_batch: Sequence[str]):
        batch_size = len(raw_batch)
        seq_str_list = raw_batch

        if self.use_atchley_factor:
            seqs, tokens, atchely_factor_feats = epitope_batch_encoding(seq_str_list, self.tokenizer, batch_size, self.max_epitope_len, True)
        else:
            seqs, tokens = epitope_batch_encoding(seq_str_list, self.tokenizer, batch_size, self.max_epitope_len, False)
            
        tokens, labels = self.tokenizer.mask_token(tokens, self.mask_prob)
        return {
            'seq': seqs,
            'token': tokens,
            'label': labels,
            'atchely_factor': atchely_factor_feats if self.use_atchley_factor else None
        }
    

class PairedCDR3BatchConverter(object):

    def __init__(self, max_cdr3_len=25, use_atchley_factor=False, mask_prob=0.15, beta_mask_prob=0.25):
        self.tokenizer = IUPACTokenizer()
        self.max_cdr3_len= max_cdr3_len
        self.use_atchley_factor = use_atchley_factor 
        self.mask_prob = mask_prob
        self.beta_mask_prob = beta_mask_prob

    def __call__(self, raw_batch: Sequence[str]):
        batch_size = len(raw_batch)
        alpha_seq, beta_seq = zip(*raw_batch)
        
        if self.use_atchley_factor:
            tokens, chain_token_mask, atchely_factor_feats = paired_cdr3_batch_encoding(alpha_seq, beta_seq, self.tokenizer, batch_size, self.max_cdr3_len, True)
        else:
            tokens, chain_token_mask = paired_cdr3_batch_encoding(alpha_seq, beta_seq, self.tokenizer, batch_size, self.max_cdr3_le, False)
            
        tokens, labels = self.tokenizer.mask_token(tokens, self.mask_prob, beta_token_mask=(chain_token_mask == 2).long(),
                                                   beta_mask_prob=self.beta_mask_prob)
        
        return {
            'token': tokens,
            'chain_token_mask': chain_token_mask,
            'label': labels,
            'atchely_factor': atchely_factor_feats if self.use_atchley_factor else None
        }
    

class PairedCDR123BatchConverter(object):
    def __init__(self, max_cdr3_len=25, use_atchley_factor=False, mask_prob=0.15, beta_mask_prob=0.25):
        self.tokenizer = IUPACTokenizer()
        self.max_cdr3_len= max_cdr3_len
        self.use_atchley_factor = use_atchley_factor
        self.mask_prob = mask_prob
        self.beta_mask_prob = beta_mask_prob
    
    def __call__(self, raw_batch: Sequence[str]):
        batch_size = len(raw_batch)
        
        if self.use_atchley_factor:
            tokens, chain_token_mask, atchely_factor_feats = paired_cdr123_batch_encoding(raw_batch, self.tokenizer, batch_size, self.max_cdr3_len, True)
        else:
            tokens, chain_token_mask = paired_cdr123_batch_encoding(raw_batch, self.tokenizer, batch_size, self.max_cdr3_len, False)
            
        tokens, labels = self.tokenizer.mask_token(tokens, self.mask_prob,
                                                   beta_token_mask=(chain_token_mask == 3) | (chain_token_mask == 6).long(),
                                                   beta_mask_prob=self.beta_mask_prob)
        
        return {
            'token': tokens,
            'chain_token_mask': chain_token_mask,
            'label': labels,
            'atchely_factor': atchely_factor_feats if self.use_atchley_factor else None
        }
        
        
class pMHCBatchConverter(object):

    # 15 for epitopes binding with MHC class I (366)
    def __init__(self, max_epitope_len=15, max_mhc_len=366):
        self.tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len
        
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}

        _, epitope_tokens = epitope_batch_encoding(data['epitope_seq'], self.tokenizer, batch_size, self.max_epitope_len)
        epi_tokens, epi_labels = self.tokenizer.mask_token(epitope_tokens)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)

        data['epitope_token'] = epi_tokens
        data['epitope_label'] = epi_labels
        data['mhc_embedding'] = mhc_embs
        data['target'] = torch.tensor(data['target'], dtype=torch.float32)
        
        return data
    

class EpitopeMHCBatchConverter(object):

    def __init__(self, max_epitope_len=15, max_mhc_len=366):
        self.tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}

        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.tokenizer, batch_size, self.max_epitope_len)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)

        data['epitope_token'] = epi_tokens
        data['mhc_embedding'] = mhc_embs
        data['pmhc_idx'] = torch.tensor(data['pmhc_idx'], dtype=torch.long)
        
        return data
    
    
class PairedCDR3pMHCBatchConverter(object):

    def __init__(self, tcr_feat_path=None,
                    max_epitope_len=15,
                    max_mhc_len=366,
                    num_pmhc=0,
                    sample_cdr3=False,
                    num_pos_pairs=5,
                    num_neg_pairs=20,
                ):
        self.epi_tokenizer = IUPACTokenizer()
        self.tcr_tokenizer = IUPACTokenizer()
        
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len
        
        self.sample_cdr3 = sample_cdr3
        if self.sample_cdr3:
            assert tcr_feat_path is not None
            self.tcr_feat_bank = load_paired_tcr_feat_bank(tcr_feat_path, num_pmhc=num_pmhc)

        self.num_pos_pairs = num_pos_pairs
        self.num_neg_pairs = num_neg_pairs

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}

        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.epi_tokenizer, batch_size, self.max_epitope_len)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)
        
        if self.sample_cdr3:
            data['pmhc_idx'] = torch.tensor(data['pmhc_idx'], dtype=torch.long)
            pos_cdr3_indices, pos_pmhc_indices, pos_labels = sample_binding_cdr3(data['pmhc_idx'], self.tcr_feat_bank['binding_mask'], self.num_pos_pairs)
            neg_cdr3_indices, neg_pmhc_indices, neg_labels = sample_non_binding_cdr3(data['pmhc_idx'], self.tcr_feat_bank['binding_mask'], self.num_neg_pairs)
            
            cdr3_indices = torch.cat([pos_cdr3_indices, neg_cdr3_indices])
            cdr3_alpha_seqs = self.tcr_feat_bank['alpha_seq'][cdr3_indices]
            cdr3_beta_seqs = self.tcr_feat_bank['beta_seq'][cdr3_indices]
            cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(cdr3_alpha_seqs, cdr3_beta_seqs, self.tcr_tokenizer, cdr3_indices.shape[0], max_cdr3_len=25)
            
            pmhc_indices = torch.cat([pos_pmhc_indices, neg_pmhc_indices])
            labels = torch.cat([pos_labels, neg_labels])
            
            raw_index = np.arange(batch_size)
            repeated_index = list(np.repeat(raw_index, self.num_pos_pairs)) + list(np.repeat(raw_index, self.num_neg_pairs))
            
            processed_data = {
                'epitope_seq': np.array(data['epitope_seq'])[repeated_index],
                'mhc_allele': np.array(data['mhc_allele'])[repeated_index],
                'cdr3_alpha_seq': cdr3_alpha_seqs,
                'cdr3_beta_seq': cdr3_beta_seqs,
                'pmhc_idx': pmhc_indices,
                'epitope_token': epi_tokens[repeated_index, :],
                'mhc_embedding': mhc_embs[repeated_index, :, :],
                'cdr3_token': cdr3_tokens,
                'chain_mask': chain_token_mask,
                'label': labels
            }
            return processed_data
            
        else:
            data['epitope_token'] = epi_tokens
            data['mhc_embedding'] = mhc_embs
            data['label'] = torch.tensor(data['label'], dtype=torch.float32)
            data['cdr3_token'], data['chain_mask'] = paired_cdr3_batch_encoding(data['cdr3_alpha_seq'], data['cdr3_beta_seq'],
                                                                                self.tcr_tokenizer, batch_size, max_cdr3_len=25)
        
            return data


class PairedCDR123pMHCBatchConverter(object):

    def __init__(self, tcr_feat_path=None,
                    max_epitope_len=15,
                    max_mhc_len=366,
                    num_pmhc=0,
                    sample_cdr3=False,
                    num_pos_pairs=5,
                    num_neg_pairs=20,
                ):
        self.epi_tokenizer = IUPACTokenizer()
        self.tcr_tokenizer = IUPACTokenizer()
        
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len
        
        self.sample_cdr3 = sample_cdr3
        if self.sample_cdr3:
            assert tcr_feat_path is not None
            self.tcr_feat_bank = load_paired_cdr123_feat_bank(tcr_feat_path, num_pmhc=num_pmhc)

        self.num_pos_pairs = num_pos_pairs
        self.num_neg_pairs = num_neg_pairs

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}

        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.epi_tokenizer, batch_size, self.max_epitope_len)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)
        
        if self.sample_cdr3:
            data['pmhc_idx'] = torch.tensor(data['pmhc_idx'], dtype=torch.long)
            pos_tcr_indices, pos_pmhc_indices, pos_labels = sample_binding_cdr3(data['pmhc_idx'], self.tcr_feat_bank['binding_mask'], self.num_pos_pairs)
            neg_tcr_indices, neg_pmhc_indices, neg_labels = sample_non_binding_cdr3(data['pmhc_idx'], self.tcr_feat_bank['binding_mask'], self.num_neg_pairs)
            
            tcr_indices = torch.cat([pos_tcr_indices, neg_tcr_indices])
            batch_data = {
                'cdr1_alpha': self.tcr_feat_bank['cdr1_alpha_seq'][tcr_indices],
                'cdr1_beta': self.tcr_feat_bank['cdr1_beta_seq'][tcr_indices],
                'cdr2_alpha': self.tcr_feat_bank['cdr2_alpha_seq'][tcr_indices],
                'cdr2_beta': self.tcr_feat_bank['cdr2_beta_seq'][tcr_indices],
                'cdr3_alpha': self.tcr_feat_bank['cdr3_alpha_seq'][tcr_indices],
                'cdr3_beta': self.tcr_feat_bank['cdr3_beta_seq'][tcr_indices]
            }
            tcr_tokens, segment_mask = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, tcr_indices.shape[0], max_cdr3_len=25)
            
            pmhc_indices = torch.cat([pos_pmhc_indices, neg_pmhc_indices])
            labels = torch.cat([pos_labels, neg_labels])
            
            raw_index = np.arange(batch_size)
            repeated_index = list(np.repeat(raw_index, self.num_pos_pairs)) + list(np.repeat(raw_index, self.num_neg_pairs))
            
            processed_data = {
                'epitope_seq': np.array(data['epitope_seq'])[repeated_index],
                'mhc_allele': np.array(data['mhc_allele'])[repeated_index],
                'cdr1_alpha_seq': batch_data['cdr1_alpha'],
                'cdr1_beta_seq': batch_data['cdr1_beta'],
                'cdr2_alpha_seq': batch_data['cdr2_alpha'],
                'cdr2_beta_seq': batch_data['cdr2_beta'],
                'cdr3_alpha_seq': batch_data['cdr3_alpha'],
                'cdr3_beta_seq': batch_data['cdr3_beta'],
                'pmhc_idx': pmhc_indices,
                'epitope_token': epi_tokens[repeated_index, :],
                'mhc_embedding': mhc_embs[repeated_index, :, :],
                'tcr_token': tcr_tokens,
                'segment_mask': segment_mask,
                'label': labels
            }
            return processed_data
            
        else:
            data['epitope_token'] = epi_tokens
            data['mhc_embedding'] = mhc_embs
            data['label'] = torch.tensor(data['label'], dtype=torch.float32)
            
            batch_data = {
                'cdr1_alpha': data['cdr1_alpha_seq'],
                'cdr1_beta': data['cdr1_beta_seq'],
                'cdr2_alpha': data['cdr2_alpha_seq'],
                'cdr2_beta': data['cdr2_beta_seq'],
                'cdr3_alpha': data['cdr3_alpha_seq'],
                'cdr3_beta': data['cdr3_beta_seq']
            }
            data['tcr_token'], data['segment_mask'] = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, batch_size, max_cdr3_len=25)
            
            return data
        

class PairedCDR3pMHCInteractBatchConverter(object):
    
    def __init__(self, max_epitope_len=15, max_mhc_len=366):
        self.epi_tokenizer = IUPACTokenizer()
        self.tcr_tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len

    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}
        
        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.epi_tokenizer, batch_size, self.max_epitope_len)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)
        tcr_tokens, chain_token_mask = paired_cdr3_batch_encoding(data['cdr3_alpha_seq'], data['cdr3_beta_seq'],
                                                    self.tcr_tokenizer, batch_size, max_cdr3_len=25)

        data['epitope_token'] = epi_tokens
        data['mhc_embedding'] = mhc_embs
        data['cdr3_token'] = tcr_tokens
        data['chain_mask'] =  chain_token_mask
        alpha_target = [target['cdr3_alpha'] for target in data['target']]
        beta_target = [target['cdr3_beta'] for target in data['target']]

        # encoding distance matrix
        dist_mat, contact_mat, alpha_masking, beta_masking = encoding_paired_dist(alpha_target, beta_target, tcr_tokens.shape[1]-1, epi_tokens.shape[1]-1)
        data['target'] = {
                    'dist': torch.tensor(dist_mat, dtype=torch.float32),
                    'contact': torch.tensor(contact_mat, dtype=torch.float32),
                    'alpha_masking': torch.tensor(alpha_masking, dtype=torch.bool),
                    'beta_masking': torch.tensor(beta_masking, dtype=torch.bool)
                }
        
        return data
    
    
class PairedCDR123pMHCInteractBatchConverter(object):
    
    def __init__(self, max_epitope_len=15, max_mhc_len=366):
        self.epi_tokenizer = IUPACTokenizer()
        self.tcr_tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len

    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}
        
        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.epi_tokenizer, batch_size, self.max_epitope_len)
        mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)
        data['epitope_token'] = epi_tokens
        data['mhc_embedding'] = mhc_embs
        
        batch_data = {
                'cdr1_alpha': data['cdr1_alpha_seq'],
                'cdr1_beta': data['cdr1_beta_seq'],
                'cdr2_alpha': data['cdr2_alpha_seq'],
                'cdr2_beta': data['cdr2_beta_seq'],
                'cdr3_alpha': data['cdr3_alpha_seq'],
                'cdr3_beta': data['cdr3_beta_seq']
        }
        data['tcr_token'], data['segment_mask'] = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, batch_size, max_cdr3_len=25)
        max_alpha_len = np.max([row.nonzero().flatten()[0].item()for row in data['segment_mask'].eq(4)]) - 1 # record the first occurrence of cdr1_beta token.
        max_beta_len = data['tcr_token'].shape[1] - max_alpha_len - 1
        
        # encoding distance matrix
        alpha_dist_mat, alpha_contact_mat, alpha_aa_mask = encoding_dist([target['alpha'] for target in data['target']], max_alpha_len, epi_tokens.shape[1]-1)
        beta_dist_mat, beta_contact_mat, beta_aa_mask = encoding_dist([target['beta'] for target in data['target']], max_beta_len, epi_tokens.shape[1]-1)
        dist_mat = np.concatenate([alpha_dist_mat, beta_dist_mat], axis=1)
        contact_mat = np.concatenate([alpha_contact_mat, beta_contact_mat], axis=1)
        aa_mask = np.concatenate([alpha_aa_mask, beta_aa_mask], axis=1)
        
        data['target'] = {
            'dist': torch.tensor(dist_mat, dtype=torch.float32),
            'contact': torch.tensor(contact_mat, dtype=torch.float32),
            'masking': torch.tensor(aa_mask, dtype=torch.bool),
        }
        return data
    

class UnlabeledBacthConverter(object):
    def __init__(self, max_epitope_len=15, max_mhc_len=366, max_cdr3_len=25, use_cdr123=False):
        self.epi_tokenizer = IUPACTokenizer()
        self.tcr_tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len
        self.max_cdr3_len = max_cdr3_len
        self.use_cdr123 = use_cdr123

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}

        _, epi_tokens = epitope_batch_encoding(data['epitope_seq'], self.epi_tokenizer, batch_size, self.max_epitope_len)
        data['epitope_token'] = epi_tokens
        
        if 'mhc_seq' in keys:
            mhc_embs = np.stack([mhc_encoding(mhc_str, self.max_mhc_len) for mhc_str in data['mhc_seq']], axis=0)
            mhc_embs = torch.tensor(mhc_embs, dtype=torch.float32)
            data['mhc_embedding'] = mhc_embs
        
        if 'cdr3_beta_seq' in keys and 'cdr3_alpha_seq' in keys:
            if 'cdr1_alpha_seq' in keys and self.use_cdr123:
                batch_data = {
                    'cdr1_alpha': data['cdr1_alpha_seq'],
                    'cdr1_beta': data['cdr1_beta_seq'],
                    'cdr2_alpha': data['cdr2_alpha_seq'],
                    'cdr2_beta': data['cdr2_beta_seq'],
                    'cdr3_alpha': data['cdr3_alpha_seq'],
                    'cdr3_beta': data['cdr3_beta_seq']
                }
                data['tcr_token'], data['segment_mask'] = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, batch_size, self.max_cdr3_len)
            else:
                data['cdr3_token'], data['chain_mask'] = paired_cdr3_batch_encoding(data['cdr3_alpha_seq'], data['cdr3_beta_seq'],
                                                                                    self.tcr_tokenizer, batch_size, self.max_cdr3_len)
                
        return data
    