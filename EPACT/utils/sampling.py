import torch
import numpy as np
from Bio import SeqIO


def sample_binding_cdr3(pmhc_idx, binding_mask, num_samples):
    cdr3_indices = []
    for idx in pmhc_idx:
        # weights = (binding_mask[:, idx] == 1).float()
        weights = binding_mask[:, idx]
        sampled_indices = torch.multinomial(weights, num_samples, replacement=True)
        cdr3_indices.append(sampled_indices)
        
    cdr3_indices = torch.cat(cdr3_indices, dim=0)
    pmhc_indices = torch.repeat_interleave(pmhc_idx, num_samples)
    labels = torch.ones_like(cdr3_indices).to(torch.float32)
    return cdr3_indices, pmhc_indices, labels


def sample_non_binding_cdr3(pmhc_idx, binding_mask, num_samples):
    cdr3_indices = []
    for idx in pmhc_idx:
        # weights = (binding_mask[:, idx] == 0).float()
        weights = 1 - binding_mask[:, idx]
        sampled_indices = torch.multinomial(weights, num_samples, replacement=False)
        cdr3_indices.append(sampled_indices)
        
    cdr3_indices = torch.cat(cdr3_indices, dim=0)
    pmhc_indices = torch.repeat_interleave(pmhc_idx, num_samples)
    labels = torch.zeros_like(cdr3_indices).to(torch.float32)
    return cdr3_indices, pmhc_indices, labels


def get_epitope_sample_weights(dataset, tf_idf=True):
    pmhc_counts = np.array([data['pmhc_count'] for data in dataset])
    num_binding_pmhc = np.sum(pmhc_counts)
    pmhc_freq = pmhc_counts / num_binding_pmhc
    if tf_idf:
        weights = pmhc_freq * np.log2(num_binding_pmhc / pmhc_counts)
        return weights
    else:
        return pmhc_freq


def get_epitope_idx_from_fasta(dataset, fasta):
    records = list(SeqIO.parse(fasta, 'fasta'))
    epitope_list = [record.seq for record in records]
    indices = [idx for idx, data in enumerate(dataset) if data['epitope_seq'] in epitope_list]
    return indices
