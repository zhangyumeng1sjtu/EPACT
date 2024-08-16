import os
import pickle
import json

from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from ..utils.encoding import hla_allele_to_seq


class EpitopeDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.epitopes = self.load_data(self.data_path)

    def load_data(self, path):
        return [line.strip() for line in open(path, 'r')]

    def __getitem__(self, idx):
        seq = self.epitopes[idx]
        return seq

    def __len__(self):
        return len(self.epitopes)
    
    
class PairedCDR3Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.cdr3_alpha, self.cdr3_beta = self.load_data(self.data_path)
        
    def load_data(self, path):
        data = pd.read_csv(path)
        return np.array(data['cdr3a']), np.array(data['cdr3b'])
    
    def __getitem__(self, idx):
        alpha_seq = self.cdr3_alpha[idx]
        beta_seq = self.cdr3_beta[idx]
        return alpha_seq, beta_seq

    def __len__(self):
        return len(self.cdr3_beta)
    

class PairedCDR123Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = self.load_data(self.data_path)
        
    def load_data(self, path):
        data = pd.read_csv(path)
        return {
            'cdr1_alpha': data['cdr1a'],
            'cdr1_beta': data['cdr1b'],
            'cdr2_alpha': data['cdr2a'],
            'cdr2_beta': data['cdr2b'],
            'cdr3_alpha': data['cdr3a'],
            'cdr3_beta': data['cdr3b'],
        }
    
    def __getitem__(self, idx):
        return {key:value[idx] for key, value in self.data.items()}

    def __len__(self):
        return len(self.data['cdr3_beta'])
    
    
class pMHCDataset(Dataset):
    
    def __init__(self, data_path, hla_lib_path, mhc_pseudo_pos=None):
        super().__init__()
        self.data_path = data_path
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data()

    def load_data(self):
        rawdata = pd.read_csv(self.data_path)       
        epitopes = rawdata['Epitope.peptide']
        mhc_alleles = rawdata['MHC']
        targets = rawdata['Target']
        
        data = {
            'epitope_seq': np.array(epitopes),
            'mhc_allele': np.array(mhc_alleles),
            'target': np.array(targets)
        }
        return data
        
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.mhc_pseudo_pos])
        return mhc_seq
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data
        
    def __len__(self):
        return len(self.data['target'])
    
    
class EpitopeMHCDataset(Dataset):
    
    def __init__(self, data_path, hla_lib_path, mhc_pseudo_pos=None):
        super().__init__()
        self.data_path = data_path
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data()

    def load_data(self):
        rawdata = pd.read_table(self.data_path)       
        epitopes = rawdata.iloc[:, 0]
        mhc_alleles = rawdata.iloc[:, 1]
        pmhc_counts = rawdata.iloc[:, 2]
        pmhc_idxes = rawdata.iloc[:, 3]
        
        data = {
            'epitope_seq': np.array(epitopes),
            'mhc_allele': np.array(mhc_alleles),
            'pmhc_count': np.array(pmhc_counts),
            'pmhc_idx': np.array(pmhc_idxes)
        }
        return data
        
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        if mhc_seq is None:
            print(mhc_allele)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.mhc_pseudo_pos])
        return mhc_seq
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data
        
    def __len__(self):
        return len(self.data['pmhc_idx'])
    
    
class PairedTCRpMHCDataset(Dataset):

    def __init__(self, data_path, hla_lib_path, mhc_pseudo_pos=None, use_cdr123=False):
        super().__init__()
        self.data_path = data_path
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data(use_cdr123)

    def load_data(self, use_cdr123):
        rawdata = pd.read_csv(self.data_path)       
        epitopes = rawdata['Epitope.peptide']
        mhc_alleles = rawdata['MHC']
        cdr3_alpha_seqs = rawdata['CDR3.alpha.aa']
        cdr3_beta_seqs = rawdata['CDR3.beta.aa']
        labels = rawdata['Target']
        data = {
            'cdr3_alpha_seq': np.array(cdr3_alpha_seqs),
            'cdr3_beta_seq': np.array(cdr3_beta_seqs),
            'epitope_seq': np.array(epitopes),
            'mhc_allele': np.array(mhc_alleles),
            'label': np.array(labels)
        }
        if use_cdr123:
            data['cdr1_alpha_seq'] = np.array(rawdata['CDR1.alpha.aa'])
            data['cdr1_beta_seq'] = np.array(rawdata['CDR1.beta.aa'])
            data['cdr2_alpha_seq'] = np.array(rawdata['CDR2.alpha.aa'])
            data['cdr2_beta_seq'] = np.array(rawdata['CDR2.beta.aa'])
            
        return data
    
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.mhc_pseudo_pos])
        return mhc_seq

    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data

    def __len__(self):
        return len(self.data['label'])
    
    
class PairedTCRpMHCInteractDataset(Dataset):
    def __init__(self, data_path, pickle_path, hla_lib_path, mhc_pseudo_pos=None, use_cdr123=False):
        super().__init__()
        self.data_path = data_path
        self.pickle_path = pickle_path
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data(use_cdr123)
        
    def load_data(self, use_cdr123):
        rawdata = pd.read_csv(self.data_path)       
        epitopes = rawdata['Epitope.peptide']
        mhc_alleles = rawdata['MHC']
        cdr3_alpha_seqs = rawdata['CDR3.alpha.aa']
        cdr3_beta_seqs = rawdata['CDR3.beta.aa']
        
        targets = []
        for i in range(len(rawdata)):
            pdb_id = rawdata.iloc[i, 0]
            pickle_path = os.path.join(self.pickle_path, f'{pdb_id}.pkl')
            with open(pickle_path, 'rb') as f:
                targets.append(pickle.load(f))

        data = {
            'cdr3_alpha_seq': np.array(cdr3_alpha_seqs),
            'cdr3_beta_seq': np.array(cdr3_beta_seqs),
            'epitope_seq': np.array(epitopes),
            'mhc_allele': np.array(mhc_alleles),
            'target': targets
        }
        if use_cdr123:
            data['cdr1_alpha_seq'] = np.array(rawdata['CDR1.alpha.aa'])
            data['cdr1_beta_seq'] = np.array(rawdata['CDR1.beta.aa'])
            data['cdr2_alpha_seq'] = np.array(rawdata['CDR2.alpha.aa'])
            data['cdr2_beta_seq'] = np.array(rawdata['CDR2.beta.aa'])
            
        return data
    
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.mhc_pseudo_pos])
        return mhc_seq
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data
        
    def __len__(self):
        return len(self.data['epitope_seq'])


class UnlabeledDataset(Dataset):
    def __init__(self, data_path, hla_lib_path, mhc_pseudo_pos=None):
        super().__init__()
        self.data_path = data_path
        self.mhc_pseudo_pos = mhc_pseudo_pos
        with open(hla_lib_path, 'r') as f:
            self.hla_library = json.load(f)
        self.data = self.load_data()

    def load_data(self):
        rawdata = pd.read_csv(self.data_path)
        epitopes = rawdata['Epitope.peptide']
        data = {'epitope_seq': np.array(epitopes)}
        
        if 'MHC' in rawdata.columns:
            mhc_alleles = rawdata['MHC']
            data['mhc_allele'] = np.array(mhc_alleles)
        
        if 'CDR1.alpha.aa' in rawdata.columns:
            cdr1_alpha_seqs = rawdata['CDR1.alpha.aa']
            cdr2_alpha_seqs = rawdata['CDR2.alpha.aa']
            data['cdr1_alpha_seq'] = np.array(cdr1_alpha_seqs)
            data['cdr2_alpha_seq'] = np.array(cdr2_alpha_seqs)
        
        if 'CDR1.beta.aa' in rawdata.columns:
            cdr1_beta_seqs = rawdata['CDR1.beta.aa']
            cdr2_beta_seqs = rawdata['CDR2.beta.aa']
            data['cdr1_beta_seq'] = np.array(cdr1_beta_seqs)
            data['cdr2_beta_seq'] = np.array(cdr2_beta_seqs)    

        if 'CDR3.alpha.aa' in rawdata.columns:
            cdr3_alpha_seqs = rawdata['CDR3.alpha.aa']
            data['cdr3_alpha_seq'] = np.array(cdr3_alpha_seqs)
            
        if 'CDR3.beta.aa' in rawdata.columns:
            cdr3_beta_seqs = rawdata['CDR3.beta.aa']
            data['cdr3_beta_seq'] = np.array(cdr3_beta_seqs)
            
        return data
        
    def transform_mhc_allele(self, mhc_allele):
        mhc_seq = hla_allele_to_seq(mhc_allele, self.hla_library)
        assert mhc_seq is not None
        if self.mhc_pseudo_pos is not None:
            mhc_seq = ''.join([mhc_seq[i] if len(mhc_seq) > i else 'X' for i in self.mhc_pseudo_pos])
        return mhc_seq
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        if 'mhc_allele' in data.keys():
            data['mhc_seq'] = self.transform_mhc_allele(data['mhc_allele'])
        return data
        
    def __len__(self):
        return len(self.data['epitope_seq'])
    