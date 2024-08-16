import os
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Sequence
import json

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F
import pandas as pd
import numpy as np

from EPACT.utils import load_config, set_seed
from EPACT.utils.encoding import hla_allele_to_seq, epitope_batch_encoding, paired_cdr3_batch_encoding, paired_cdr123_batch_encoding, mhc_encoding
from EPACT.utils.tokenizer import IUPACTokenizer
from EPACT.model import PairedCDR3pMHCCoembeddingModel, PairedCDR123pMHCCoembeddingModel
from EPACT.dataset import UnlabeledDataset, UnlabeledBacthConverter


# define pMHC and TCR(CDR3) dataset
class pMHCDataset(Dataset):
    def __init__(self, rawdata, hla_lib_path, mhc_pseudo_pos=None):
        super().__init__()
        self.rawdata = rawdata
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data()

    def load_data(self):
        epitopes = self.rawdata['Epitope.peptide']
        data = {'epitope_seq': np.array(epitopes)}
        
        if 'MHC' in self.rawdata.columns:
            mhc_alleles = self.rawdata['MHC']
            data['mhc_allele'] = np.array(mhc_alleles)
        return data
        
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.pmhc_pseudo_pos])
        return mhc_seq
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        if 'mhc_allele' in data.keys():
            data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data
        
    def __len__(self):
        return len(self.data['epitope_seq'])
    
    
class pMHCBacthConverter(object):
    def __init__(self, max_epitope_len=15, max_mhc_len=366):
        self.epi_tokenizer = IUPACTokenizer()
        self.max_epitope_len = max_epitope_len
        self.max_mhc_len = max_mhc_len

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
                
        return data
  
  
class TCRDataset(Dataset):
    def __init__(self, rawdata, use_cdr123=False):
        super().__init__()
        self.rawdata = rawdata
        self.use_cdr123 = use_cdr123
        self.data = self.load_data()

    def load_data(self):
        data = {}

        if 'cdr3a' in self.rawdata.columns:
            cdr3_alpha_seqs = self.rawdata['cdr3a']
            data['cdr3_alpha_seq'] = np.array(cdr3_alpha_seqs)
            
        if 'cdr3b' in self.rawdata.columns:
            cdr3_beta_seqs = self.rawdata['cdr3b']
            data['cdr3_beta_seq'] = np.array(cdr3_beta_seqs)
        
        if 'cdr1a' in self.rawdata.columns and self.use_cdr123:
            data['cdr1_alpha_seq'] = np.array(self.rawdata['cdr1a'])
            data['cdr2_alpha_seq'] = np.array(self.rawdata['cdr2a'])
            data['cdr1_beta_seq'] = np.array(self.rawdata['cdr1b'])
            data['cdr2_beta_seq'] = np.array(self.rawdata['cdr2b'])  
            
        return data
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        return data
        
    def __len__(self):
        return len(self.data['cdr3_beta_seq'])

    
class TCRBacthConverter(object):
    def __init__(self, use_cdr123=False):
        self.tcr_tokenizer = IUPACTokenizer()
        self.use_cdr123 = use_cdr123

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        keys = raw_batch[0].keys()
        data = {key: [d[key] for d in raw_batch] for key in keys}
        
        if 'cdr3_beta_seq' in keys and 'cdr3_alpha_seq' in keys:
            if 'cdr1_alpha_seq' in keys and self.use_cdr123:
                batch_data = {
                    'cdr1_alpha': data['cdr1_alpha_seq'],
                    'cdr2_alpha': data['cdr2_alpha_seq'],
                    'cdr1_beta': data['cdr1_beta_seq'],
                    'cdr2_beta': data['cdr2_beta_seq'],
                    'cdr3_alpha': data['cdr3_alpha_seq'],
                    'cdr3_beta': data['cdr3_beta_seq']
                }
                data['tcr_token'], data['segment_mask'] = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, batch_size, max_cdr3_len=25)
            else:
                data['cdr3_token'], data['chain_mask'] = paired_cdr3_batch_encoding(data['cdr3_alpha_seq'], data['cdr3_beta_seq'], self.tcr_tokenizer, batch_size, max_cdr3_len=25)
                
        return data    


def percantage_rank(value, sorted_array):
    mask = (sorted_array - value) >= 0
    if np.any(mask):
        return np.argmax(mask) * 100 / len(sorted_array)
    else:
        return 100
    

def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    
    set_seed(config.training.seed)
    device = torch.device(f"cuda:{config.training.gpu_device}" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    if config.model.mhc_seq_len == 34:
            pseudo_seq_pos = [7, 9, 24, 45, 59, 62, 63, 66, 67, 79, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97,
                            99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171]  # NetMHCpan pseudosequence
    elif config.model.mhc_seq_len == 30:
        pseudo_seq_pos = [2, 9, 10, 33, 35, 48, 86, 90, 91, 93, 94, 95, 100, 101, 105, 119, 121,
                            138, 140, 176, 180, 182, 187, 206, 316, 317, 319, 320, 329, 339]  # BigMHC pseudosequence
    else:
        pseudo_seq_pos = None
    
    # predict TCR-pMHC binding score
    print("Predicting Binding scores")
    dataset = UnlabeledDataset(data_path = args.input_data_path,
                                hla_lib_path = config.data.hla_lib_path,
                                mhc_pseudo_pos = pseudo_seq_pos)
    test_loader = DataLoader(
        dataset = dataset, batch_size = config.training.test_batch_size, num_workers = config.training.num_workers,
        collate_fn = UnlabeledBacthConverter(max_mhc_len = config.model.mhc_seq_len, use_cdr123=config.data.use_cdr123),
        shuffle = False
    )

    if config.data.use_cdr123:
        model = PairedCDR123pMHCCoembeddingModel(**config.model)
    else:
        model = PairedCDR3pMHCCoembeddingModel(**config.model)    
    model.load_state_dict(torch.load(args.model_location, map_location='cpu'))
    model.to(device)
    
    preds, sims, epi, mhc, cdr3_alpha, cdr3_beta = [], [], [], [], [], []
    if config.data.use_cdr123:
        cdr1_alpha, cdr2_alpha, cdr1_beta, cdr2_beta = [], [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                                
            epi_embed, epi_padding_mask = model.compute_pmhc_feat(batch)
            if config.data.use_cdr123:
                tcr_embed, tcr_padding_mask = model.compute_tcr_feat(batch['tcr_token'], batch['segment_mask'])
            else:
                tcr_embed, tcr_padding_mask, _ = model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
            
            out = model.forward(
                epi_feat = epi_embed,
                tcr_feat = tcr_embed,
                epi_padding_mask = epi_padding_mask,
                tcr_padding_mask = tcr_padding_mask
            )
            logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
            preds.append(logits.detach().cpu().numpy())
            sims.append(similarity.detach().cpu().numpy())
            epi.extend(batch['epitope_seq'])
            mhc.extend(batch['mhc_allele'])
            cdr3_alpha.extend(batch['cdr3_alpha_seq'])
            cdr3_beta.extend(batch['cdr3_beta_seq'])
            
            if config.data.use_cdr123:
                cdr1_alpha.extend(batch['cdr1_alpha_seq'])
                cdr1_beta.extend(batch['cdr1_beta_seq'])
                cdr2_alpha.extend(batch['cdr2_alpha_seq'])
                cdr2_beta.extend(batch['cdr2_beta_seq'])
                
        preds = np.concatenate(preds)
        sims = np.concatenate(sims)
        
        if config.data.use_cdr123:
            res = pd.DataFrame({'Epitope.peptide': epi, 'MHC': mhc, 'CDR1.alpha.aa': cdr1_alpha, 'CDR1.beta.aa': cdr1_beta,
                                 'CDR2.alpha.aa': cdr2_alpha, 'CDR2.beta.aa': cdr2_beta, 'CDR3.alpha.aa': cdr3_alpha,
                                 'CDR3.beta.aa': cdr3_beta, 'Pred': preds, 'Similarity': sims})
        else:
            res = pd.DataFrame({'Epitope.peptide': epi, 'MHC': mhc,
                            'CDR3.alpha.aa': cdr3_alpha, 'CDR3.beta.aa': cdr3_beta, 'Pred': preds, 'Similarity': sims})
      
      
    print("Predicting Background scores")
    pmhc = res[['Epitope.peptide', 'MHC']].value_counts().reset_index()
    pmhc_dataset = pMHCDataset(pmhc, hla_lib_path = config.data.hla_lib_path, mhc_pseudo_pos = pseudo_seq_pos)
    pmhc_loader = DataLoader(
        dataset = pmhc_dataset, batch_size = 1, num_workers = config.training.num_workers,
        collate_fn = pMHCBacthConverter(max_mhc_len = config.model.mhc_seq_len),
        shuffle = False
    )
    
    tcr_bg_dataset = TCRDataset(rawdata = pd.read_csv(args.bg_tcr_path), use_cdr123 = config.data.use_cdr123)
    tcr_bg_loader = DataLoader(
        dataset = tcr_bg_dataset, sampler = RandomSampler(tcr_bg_dataset, num_samples=args.num_bg_tcrs), batch_size = 1000,
        num_workers = config.training.num_workers, collate_fn = TCRBacthConverter(use_cdr123 = config.data.use_cdr123),
    )
    
    bg_score_dict = {}
    bg_similarity_dict = {}
    
    with torch.no_grad():
        for pmhc_batch in pmhc_loader:
            epitope, mhc = pmhc_batch['epitope_seq'][0], pmhc_batch['mhc_allele'][0].replace('HLA-', '')
            print('-'.join([epitope, mhc]))
            for key, value in pmhc_batch.items():
                if isinstance(value, torch.Tensor):
                    pmhc_batch[key] = value.to(device)
            
            epi_embed, epi_padding_mask = model.compute_pmhc_feat(pmhc_batch)
            pmhc_embedding, pmhc_projection = model.compute_pmhc_embedding(epi_embed, epi_padding_mask)
            pmhc_embedding = pmhc_embedding.repeat(1000, 1)
            
            logits_list = []
            similarity_list = []
            
            for tcr_batch in tqdm(tcr_bg_loader):
                for key, value in tcr_batch.items():
                    if isinstance(value, torch.Tensor):
                        tcr_batch[key] = value.to(device)

                if config.data.use_cdr123:
                    tcr_embed, tcr_padding_mask = model.compute_tcr_feat(tcr_batch['tcr_token'], tcr_batch['segment_mask'])
                else:
                    tcr_embed, tcr_padding_mask, _ = model.compute_tcr_feat(tcr_batch['cdr3_token'], tcr_batch['chain_mask'])
                    
                tcr_embedding, tcr_projection = model.compute_tcr_embedding(tcr_embed, tcr_padding_mask)
                logits = model.get_logits(tcr_embedding, pmhc_embedding).squeeze(-1)
                logits_list.append(logits.detach().cpu().numpy())
                
                similarity = F.cosine_similarity(pmhc_projection, tcr_projection, dim=-1)
                similarity_list.append(similarity.detach().cpu().numpy())
            
            bg_score_dict['-'.join([epitope, mhc])] = np.sort(np.concatenate(logits_list))
            bg_similarity_dict['-'.join([epitope, mhc])] = np.sort(np.concatenate(similarity_list))
            
            print('Median binding score:', np.median(bg_score_dict['-'.join([epitope, mhc])]))
            print('Median similarity:', np.median(bg_similarity_dict['-'.join([epitope, mhc])]))
    
    res['MHC'] = res['MHC'].apply(lambda x: x.replace('HLA-', ''))
    res['Pred.Rank'] = res.apply(lambda x: percantage_rank(x['Pred'], bg_score_dict['-'.join([x['Epitope.peptide'], x['MHC']])]), axis=1)
    res['Similarity.Rank'] = res.apply(lambda x: percantage_rank(x['Similarity'], bg_similarity_dict['-'.join([x['Epitope.peptide'], x['MHC']])]), axis=1)
    res.to_csv(os.path.join(args.log_dir, 'predictions.csv'), index=False)
    
    
if __name__ == '__main__':
   
    '''
    python predict_tcr_pmhc_binding_rank.py --config logs/paired-cdr123-pmhc-binding/config.yml \
                                            --log_dir <log_dir> \
                                            --input_data_path <input_data_path> \
                                            --model_location <checkpoint_path> \
                                            --bg_tcr_path <bg_tcr_repertoire_path> \
                                            --num_bg_tcrs <num_bg_tcrs>    
    '''
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config-paired-cdr123-pmhc-binding.yml')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--input_data_path', type=str, required=True)
    parser.add_argument('--model_location', type=str, required=True)
    parser.add_argument('--bg_tcr_path', type=str, required=True)
    parser.add_argument('--num_bg_tcrs', type=int, default=20000)
    
    args = parser.parse_args()
    
    main(args)           
                
    