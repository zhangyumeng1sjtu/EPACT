import random

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, median_absolute_error
from scipy.stats import pearsonr
from easydict import EasyDict
import yaml


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    

def load_paired_tcr_feat_bank(tcr_feat_path, num_pmhc):
    tcr_feat_dict = torch.load(tcr_feat_path)
    cdr3_alpha_seqs, cdr3_beta_seqs = [], []
    binding_mask = torch.zeros((len(tcr_feat_dict), num_pmhc))
    
    for i, value in enumerate(tcr_feat_dict):
        cdr3_alpha_seqs.append(value['cdr3.alpha'])
        cdr3_beta_seqs.append(value['cdr3.beta'])
        binding_mask[i, value['pmhc']] = 1

    return {
        'alpha_seq': np.array(cdr3_alpha_seqs),
        'beta_seq': np.array(cdr3_beta_seqs),
        'binding_mask': binding_mask
    }


def load_paired_cdr123_feat_bank(tcr_feat_path, num_pmhc):
    tcr_feat_dict = torch.load(tcr_feat_path)
    cdr1_alpha_seqs, cdr1_beta_seqs, cdr2_alpha_seqs, cdr2_beta_seqs, cdr3_alpha_seqs, cdr3_beta_seqs = [], [], [], [], [], []
    binding_mask = torch.zeros((len(tcr_feat_dict), num_pmhc))
    
    for i, value in enumerate(tcr_feat_dict):
        cdr1_alpha_seqs.append(value['cdr1.alpha'])
        cdr1_beta_seqs.append(value['cdr1.beta'])
        cdr2_alpha_seqs.append(value['cdr2.alpha'])
        cdr2_beta_seqs.append(value['cdr2.beta'])
        cdr3_alpha_seqs.append(value['cdr3.alpha'])
        cdr3_beta_seqs.append(value['cdr3.beta'])
        binding_mask[i, value['pmhc']] = 1
        
    return {
        'cdr1_alpha_seq': np.array(cdr1_alpha_seqs),
        'cdr1_beta_seq': np.array(cdr1_beta_seqs),
        'cdr2_alpha_seq': np.array(cdr2_alpha_seqs),
        'cdr2_beta_seq': np.array(cdr2_beta_seqs),
        'cdr3_alpha_seq': np.array(cdr3_alpha_seqs),
        'cdr3_beta_seq': np.array(cdr3_beta_seqs),
        'binding_mask': binding_mask,
    }


def get_scores_dist(y_true, y_pred, y_mask):
    coef, mae, mape = [], [], []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):

        y_true_ = np.array(y_true_)
        y_pred_ = np.array(y_pred_)
        y_mask_ = np.array(y_mask_)

        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            coef_, _ = pearsonr(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

        mae_ = median_absolute_error(y_true_, y_pred_)
        mae.append(mae_)

        mape_ = np.median(np.abs((y_true_ - y_pred_) / y_true_))
        mape.append(mape_)
        
    avg_coef = np.nanmean(coef)
    avg_mae = np.mean(mae)
    avg_mape = np.mean(mape)
    return [avg_coef, avg_mae, avg_mape], [coef, mae, mape]


def get_scores_contact(y_true, y_pred, y_mask):
    coef = []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):

        y_true_ = np.array(y_true_)
        y_pred_ = np.array(y_pred_)
        y_mask_ = np.array(y_mask_)
        
        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            coef_ = roc_auc_score(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

    avg_coef = np.nanmean(coef)
    return [avg_coef], [coef]
