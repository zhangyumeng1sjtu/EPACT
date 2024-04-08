import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from scipy.stats import pearsonr

from .base import BaseTrainer
from ..model import EpitopeMHCModel, PairedCDR3pMHCCoembeddingModel, PairedCDR123pMHCCoembeddingModel
from ..model.utils import EarlyStopping, FocalLoss
from ..utils.misc import load_paired_tcr_feat_bank, load_paired_cdr123_feat_bank
from ..utils.encoding import paired_cdr3_batch_encoding, paired_cdr123_batch_encoding


class EpitopeMHCTrainer(BaseTrainer):
    def __init__(self, config, log_dir=None):
        super(EpitopeMHCTrainer, self).__init__(config, log_dir)

        self.model = EpitopeMHCModel(**config.model)
        # Load pretrained LM or Not
        self.model.load_pretrained_epitope_model(config.training.pretrained_epitope_model)
        self.model.to(self.device)
        
        self.task = config.task  # BA or EL
        if self.task == 'BA':
            self.loss_fn = nn.MSELoss()
        elif self.task == 'EL':
            if config.training.use_focal_loss:
                self.loss_fn = FocalLoss(alpha=0.8, gamma=2)
            else:
                self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Invalid prediction task (MUST be BA or EL)')
        
        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.lambda_factor = config.training.lambda_factor

        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
    
    #TODO try a trainer w/o mlm loss
    def proceed_one_epoch(self, iterator, training=True, return_predictions=False, add_mlm_loss=False):
        avg_sup_loss = 0
        avg_mlm_loss = 0
        data_size = 0
        pred = []
        y = []
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        for batch in tqdm(iterator):
            epi_toks = batch['epitope_token'].to(self.device)
            epi_labels = batch['epitope_label'].to(self.device)
            mhc_embeds = batch['mhc_embedding'].to(self.device)
            targets = batch['target'].to(self.device)
            
            if training and add_mlm_loss:
                epi_logits, out = self.model(epi_toks, mhc_embeds)
            else:
                _, out = self.model(epi_labels, mhc_embeds)
            
            out = out.squeeze(-1)
            sup_loss = self.loss_fn(out, targets)
            
            if training:
                if add_mlm_loss: 
                    mask = (epi_toks == self.tokenizer.mask_idx)
                    mask_logits = epi_logits[mask]
                    mask_labels = epi_labels[mask]
                    weights = targets[torch.nonzero(mask, as_tuple=False)[:, 0]]
                    mlm_loss = torch.mean(self.mlm_loss_fn(mask_logits, mask_labels) * weights)
                
                    loss = sup_loss + self.lambda_factor * mlm_loss
                else:
                    loss = sup_loss

            pred.append(out.detach().cpu().numpy())
            y.append(targets.detach().cpu().numpy())

            avg_sup_loss += sup_loss.cpu().item() * len(targets)
            data_size += len(targets)
            
            if training:
                if add_mlm_loss:
                    avg_mlm_loss += mlm_loss.cpu().item() * len(targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        pred = np.concatenate(pred)
        y = np.concatenate(y)
        
        avg_sup_loss = avg_sup_loss / data_size
        avg_mlm_loss = avg_mlm_loss / data_size

        if self.task == 'BA':
            pearson_coef = pearsonr(pred, y).statistic
            metrics = {
                'sup_loss': avg_sup_loss,
                'mlm_loss': avg_mlm_loss,
                'coef': pearson_coef
            }
            if return_predictions:
                return metrics, y, pred
            return metrics
        
        elif self.task == 'EL':
            avg_acc = accuracy_score(y, (pred >= 0.5).astype(int))
            avg_f1 = f1_score(y, (pred >= 0.5).astype(int))
            avg_auprc = average_precision_score(y, pred)
            metrics = {
                'sup_loss': avg_sup_loss,
                'mlm_loss': avg_mlm_loss,
                'acc': avg_acc,
                'f1': avg_f1,
                'auprc': avg_auprc
            }
            if return_predictions:
                return metrics, y, pred
            return metrics
    
    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True,  add_mlm_loss=True)

    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        with torch.no_grad():
            return self.proceed_one_epoch(eval_loader, training=False, return_predictions=return_predictions)
         
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(
                f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            if self.task == 'BA':
                self.logger.info(f'Train Supervised Loss: {train_metrics["sup_loss"]:.4f} | MLM Loss: {train_metrics["mlm_loss"]:.3f}')
                self.logger.info(f'Train Coef: {train_metrics["coef"]:.3f}')
                self.logger.info(f'Valid Supervised Loss: {val_metrics["sup_loss"]:.4f}')
                self.logger.info(f'Valid Coef: {val_metrics["coef"]:.3f}')
            elif self.task == 'EL':
                self.logger.info(f'Train Supervised Loss: {train_metrics["sup_loss"]:.4f} | MLM Loss: {train_metrics["mlm_loss"]:.3f}')
                self.logger.info(f'Train ACC: {train_metrics["acc"]*100:.2f}% | F1: {train_metrics["f1"]:.3f} | AUPRC: {train_metrics["auprc"]:.3f}')
                self.logger.info(f'Valid Supervised Loss: {val_metrics["sup_loss"]:.4f}')
                self.logger.info(f'Valid ACC: {val_metrics["acc"]*100:.2f}% | F1: {val_metrics["f1"]:.3f} | AUPRC: {val_metrics["auprc"]:.3f}')
                
            if self.task == 'BA':
                for key in ['sup_loss', 'mlm_loss', 'coef']:
                    self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                    self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                self.early_stopping(val_metrics["coef"], self.model, goal="maximize")

            elif self.task == 'EL':
                for key in ['sup_loss', 'mlm_loss', 'acc', 'f1', 'auprc']:
                    self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                    self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                self.early_stopping(val_metrics["auprc"], self.model, goal="maximize")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
        self.writer.close()

    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        if self.task == 'BA':
            test_metrics, y, pred = self.evaluate_one_epoch(test_loader, return_predictions=True)
            self.logger.info(f'Test Supervised Loss: {test_metrics["sup_loss"]:.4f}')
            self.logger.info(f'Test Coef: {test_metrics["coef"]:.3f}')
        elif self.task == 'EL':
            test_metrics, y, pred = self.evaluate_one_epoch(test_loader, return_predictions=True)
            self.logger.info(f'Test Supervised Loss: {test_metrics["sup_loss"]:.4f}')
            self.logger.info(f'Test ACC: {test_metrics["acc"]*100:.2f}% | F1: {test_metrics["f1"]:.3f} | AUPRC: {test_metrics["auprc"]:.3f}')

        res = pd.DataFrame({'Target': y, 'Pred': pred})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)

    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                epi_toks = batch['epitope_token'].to(self.device)
                mhc_embeds = batch['mhc_embedding'].to(self.device)
                _, out = self.model(epi_toks, mhc_embeds)

                pred.append(out.squeeze(-1).detach().cpu().numpy())
        pred = np.concatenate(pred)
        df = pd.DataFrame({'Pred': pred})
        df.to_csv(os.path.join(self.log_dir, 'predictions.csv'), index=False)


class PairedCDR3pMHCCoembeddingTrainer(BaseTrainer):
    
    def __init__(self, config, log_dir=None):
        super(PairedCDR3pMHCCoembeddingTrainer, self).__init__(config, log_dir)
        
        self.model = PairedCDR3pMHCCoembeddingModel(**config.model)
        self.model.load_pretrained_pmhc_model(config.training.pretrained_pmhc_model)
        self.model.load_pretrained_tcr_model(config.training.pretrained_tcr_model)
        self.model.to(self.device)
        
        param_finetune_list = []
        param_default_list = []
        for name, param in self.model.named_parameters():
            if name.startswith('tcr_model'):
                param_finetune_list.append(param)
            elif name.startswith('pmhc_model.epitope_model'):
                param_finetune_list.append(param)
            elif name.startswith('pmhc_model.mhc_model'):
                param_finetune_list.append(param)
            else:
                param_default_list.append(param)
                
        self.optimizer, self.scheduler = self.configure_optimizer([
            {'params': param_finetune_list, 'lr': config.training.lr / 10},
            {'params': param_default_list, 'lr': config.training.lr},    
        ], config.training)
        
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        self.num_neg_pairs = config.training.non_binding_ratio # default: 5
        self.contrasruce_loss_type = config.training.contrasruce_loss_type
        self.contrastive_loss_coef = config.training.contrastive_loss_coef
        
        self.tcr_tokenizer = self.model.tokenizer
        self.tcr_feat_path = config.data.train_tcr_feat_path
        self.pmhc_bank = pd.read_table(config.data.train_pmhc_path)
        
        self.temperature = config.training.temperature
        self.margin = config.training.margin
        self.embed_hid_dim = config.model.embed_hid_dim
        
    def generate_nonbinding_tcr(self, batch):
        batch_size = len(batch['epitope_seq'])
        cdr3_indices = []
        for epi, mhc in zip(batch['epitope_seq'], batch['mhc_allele']):
            pmhc_idx = self.pmhc_bank.loc[(self.pmhc_bank['Epitope.peptide'] == epi) & (self.pmhc_bank['MHC'] == mhc), 'pmhc_idx'].values[0]
            weights = (self.tcr_feat_bank['training_mask'][:, pmhc_idx] == 0).float() 
            sampled_indices = torch.multinomial(weights, self.num_neg_pairs, replacement=False)
            cdr3_indices.append(sampled_indices)
            
        cdr3_indices = torch.cat(cdr3_indices, dim=0)
        non_binding_cdr3_alpha = self.tcr_feat_bank['alpha_seq'][cdr3_indices].tolist()
        non_binding_cdr3_beta = self.tcr_feat_bank['beta_seq'][cdr3_indices].tolist()
        
        binding_cdr3_alpha, binding_cdr3_beta = batch['cdr3_alpha_seq'], batch['cdr3_beta_seq']

        batch['cdr3_token'], batch['chain_mask'] = paired_cdr3_batch_encoding(binding_cdr3_alpha + non_binding_cdr3_alpha,
                                                                   binding_cdr3_beta + non_binding_cdr3_beta, self.tcr_tokenizer, batch_size * 6, max_cdr3_len=25)
        
        raw_index = np.arange(batch_size)
        repeated_index = list(raw_index) + list(np.repeat(raw_index, 5))
        batch['epitope_token'] = batch['epitope_token'][repeated_index, :]
        batch['mhc_embedding'] = batch['mhc_embedding'][repeated_index, :, :]
        batch['label'] = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 5)])

        return batch
               
    @staticmethod
    def compute_contrastive_loss(anchor, positive, negative, temperature=1.0):
        """
            anchor: B, E; positive B, E; negative: 5B, E
        """
        logits_pos = F.cosine_similarity(anchor, positive) # B
        negative = negative.view(-1, 5, anchor.size(1)) # B, 5, E
        logits_neg = torch.stack([F.cosine_similarity(anchor, neg) for neg in negative.transpose(0, 1)], dim=1) # B, 5
        
        numerator = torch.exp(logits_pos / temperature)
        denominator = numerator + torch.exp(logits_neg / temperature).sum(1)
        
        loss = - torch.log(numerator / denominator)
        return loss.mean()
    
    @staticmethod
    def compute_triplet_loss(anchor, positive, negative, margin=0.3):
        """
            anchor: B, E; positive B, E; negative: 5B, E
        """
        dist_pos = 1 - F.cosine_similarity(anchor, positive) # B
        negative = negative.view(-1, 5, anchor.size(1)) # B, 5, E
        dist_neg = torch.stack([1 - F.cosine_similarity(anchor, neg) for neg in negative.transpose(0, 1)], dim=1) # B, 5
        
        loss = F.relu(dist_pos.unsqueeze(1) - dist_neg + margin)
        return loss.mean()
    
    @staticmethod
    def compute_clf_metrics(y, pred, epitopes):
        avg_acc = accuracy_score(y, (pred >= 0.5).astype(int))
        avg_f1 = f1_score(y, (pred >= 0.5).astype(int))
        avg_auroc = roc_auc_score(y, pred)
        avg_auprc = average_precision_score(y, pred)
        
        res = pd.DataFrame({'epitope': epitopes, 'y': y, 'pred': pred})
        avg_epi_auroc = res.groupby('epitope').apply(lambda x: roc_auc_score(x['y'], x['pred'])).mean()
        avg_epi_auprc = res.groupby('epitope').apply(lambda x: average_precision_score(x['y'], x['pred'])).mean()
        
        return {
            'acc': avg_acc,
            'f1': avg_f1,
            'auc': avg_auroc,
            'aupr': avg_auprc,
            'auc_epi': avg_epi_auroc,
            'aupr_epi': avg_epi_auprc
        }
        
    def proceed_one_epoch(self, iterator, training=True, return_predictions=False):  
        avg_loss, data_size = 0, 0            
        pred, y, epitope_seqs = [], [], []
        
        if training:
            self.model.train()
            avg_bce_loss, avg_contrastive_loss = 0, 0
        else:
            self.model.eval()
            
        for batch in tqdm(iterator):
            
            if training:
                batch = self.generate_nonbinding_tcr(batch) # sample nonbinding tcrs from the feature bank
            
            for key, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[key] = item.to(self.device)
                    
            if training:     
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                cdr3_embed, cdr3_padding_mask, _ = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
                    
                out = self.model.forward(
                    epi_feat = epi_embed,
                    tcr_feat = cdr3_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr3_padding_mask
                )
                logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                tcr_projections, pmhc_projections = out['projection']
                labels = batch['label']
                batch_size = labels.shape[0] // 6
                anchor_projections = pmhc_projections[:batch_size, :]
                positive_projections = tcr_projections[:batch_size, :]
                negative_projections = tcr_projections[batch_size:, :]
        
                bce_loss = F.binary_cross_entropy(logits, labels)
                if self.contrasruce_loss_type == 'simclr':
                    contrastive_loss = self.compute_contrastive_loss(anchor_projections, positive_projections, negative_projections, temperature=self.temperature)
                elif self.contrasruce_loss_type == 'triplet':
                    contrastive_loss = self.compute_triplet_loss(anchor_projections, positive_projections, negative_projections, margin=self.margin)
                elif self.contrasruce_loss_type == 'simclr+triplet':
                    contrastive_loss = self.compute_contrastive_loss(anchor_projections, positive_projections, negative_projections, temperature=self.temperature) + self.compute_triplet_loss(anchor_projections, positive_projections, negative_projections, labels, margin=self.margin)
                    
                loss = bce_loss + self.contrastive_loss_coef * contrastive_loss
                epitope_seqs.extend(batch['epitope_seq'] + [seq for seq in batch['epitope_seq'] for _ in range(self.num_neg_pairs)])        
            
            else:
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                cdr3_embed, cdr3_padding_mask, _ = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
                    
                out = self.model.forward(
                    epi_feat = epi_embed,
                    tcr_feat = cdr3_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr3_padding_mask
                )
                logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                labels = batch['label']
                loss = F.binary_cross_entropy(logits, labels)
                epitope_seqs.extend(batch['epitope_seq'])
                
            pred.append(logits.detach().cpu().numpy())
            y.append(labels.detach().cpu().numpy())
            
            avg_loss += loss.item() * len(batch)
            data_size += len(batch)
            
            if training:
                avg_bce_loss += bce_loss.item() * len(batch)
                avg_contrastive_loss += contrastive_loss.item() * len(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)
        
        metrics = self.compute_clf_metrics(y, pred, epitope_seqs)
        metrics['loss'] = avg_loss / data_size
        metrics['bce_loss'] = avg_bce_loss / data_size if training else avg_loss / data_size
        metrics['contrastive_loss'] = avg_contrastive_loss / data_size if training else 0.0
        
        if return_predictions:
            return metrics, y, pred
        else:
            return metrics
    
    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)
    
    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        return self.proceed_one_epoch(eval_loader, training=False, return_predictions=return_predictions)
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)
        self.tcr_feat_bank = load_paired_tcr_feat_bank(self.tcr_feat_path, num_pmhc=249)
        self.tcr_feat_bank['training_mask'] = self.tcr_feat_bank['binding_mask'].clone()
        
        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            
            self.logger.info(f'Train Loss: {train_metrics["loss"]:.3f} | BCE Loss: {train_metrics["bce_loss"]:.3f} | Contrastive Loss: {train_metrics["contrastive_loss"]:.3f}')
            self.logger.info(f'Train ACC: {train_metrics["acc"]*100:.2f}% | F1: {train_metrics["f1"]:.3f} | AUROC {train_metrics["auc"]:.3f} | AUPRC: {train_metrics["aupr"]:.3f}')
            self.logger.info(f'Train AVG AUROC: {train_metrics["auc_epi"]:.3f} | AVG AUPRC: {train_metrics["aupr_epi"]:.3f}')
            
            self.logger.info(f'Valid Loss: {val_metrics["loss"]:.3f} | BCE Loss: {val_metrics["bce_loss"]:.3f} | Contrastive Loss: {val_metrics["contrastive_loss"]:.3f}')
            self.logger.info(f'Valid ACC: {val_metrics["acc"]*100:.2f}% | F1: {val_metrics["f1"]:.3f} | AUROC {val_metrics["auc"]:.3f} | AUPRC: {val_metrics["aupr"]:.3f}')
            self.logger.info(f'Valid AVG AUROC: {val_metrics["auc_epi"]:.3f} | AVG AUPRC: {val_metrics["aupr_epi"]:.3f}')
            
            for key, value in train_metrics.items(): 
                self.writer.add_scalar(f'Train/{key}', value, epoch+1)
            for key, value in val_metrics.items(): 
                self.writer.add_scalar(f'Valid/{key}', value, epoch+1)

            if epoch >= 19:
                self.early_stopping(val_metrics["auc"], self.model, goal="maximize")
                # self.early_stopping(val_metrics["aupr"], self.model, goal="maximize")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break

        self.writer.close()
        
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        test_metrics, y, pred = self.evaluate_one_epoch(test_loader, return_predictions=True)
        self.logger.info(f'Test Loss: {test_metrics["loss"]:.3f} | BCE Loss: {test_metrics["bce_loss"]:.3f} | Contrastive Loss: {test_metrics["contrastive_loss"]:.3f}')
        self.logger.info(f'Test ACC: {test_metrics["acc"]*100:.2f}% | F1: {test_metrics["f1"]:.3f} | AUROC {test_metrics["auc"]:.3f} | AUPRC: {test_metrics["aupr"]:.3f}')
        self.logger.info(f'Test AVG AUROC: {test_metrics["auc_epi"]:.3f} | AVG AUPRC: {test_metrics["aupr_epi"]:.3f}')
        for key, value in test_metrics.items(): 
            self.writer.add_scalar(f'Test/{key}', value)
        
        res = pd.DataFrame({'Target': y, 'Pred': pred})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)
        
    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, map_location='cpu'), strict=False)
        preds, sims, epi, mhc, cdr3_alpha, cdr3_beta = [], [], [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                                 
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                cdr3_embed, cdr3_padding_mask, _ = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
                
                out = self.model.forward(
                    epi_feat = epi_embed,
                    tcr_feat = cdr3_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr3_padding_mask
                )
                logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                preds.append(logits.detach().cpu().numpy())
                sims.append(similarity.detach().cpu().numpy())
                epi.extend(batch['epitope_seq'])
                mhc.extend(batch['mhc_allele'])
                cdr3_alpha.extend(batch['cdr3_alpha_seq'])
                cdr3_beta.extend(batch['cdr3_beta_seq'])
                
            preds = np.concatenate(preds)
            sims = np.concatenate(sims)
            df = pd.DataFrame({'Epitope.peptide': epi, 'MHC': mhc,
                            'CDR3.alpha.aa': cdr3_alpha, 'CDR3.beta.aa': cdr3_beta, 'Pred': preds, 'Similarity': sims})
            df.to_csv(os.path.join(self.log_dir, 'predictions.csv'), index=False)   
            
    def predict_embeddings(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location), strict=False)
        pmhc_embeddings, cdr3_embeddings = {}, {}
        if self.model.agg == 'attn':
            pmhc_attn_dict, tcr_attn_dict = {}, {}
            
        self.model.eval()
        for batch in tqdm(data_loader):
            for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                                
            epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
            cdr3_embed, cdr3_padding_mask, _ = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
            
            out = self.model.forward(
                epi_feat = epi_embed,
                tcr_feat = cdr3_embed,
                epi_padding_mask = epi_padding_mask,
                tcr_padding_mask = cdr3_padding_mask
            )
            tcr_projection, pmhc_projection = out['projection']
            tcr_projection = tcr_projection.detach().cpu().numpy()
            pmhc_projection = pmhc_projection.detach().cpu().numpy()
            
            if self.model.agg == 'attn':
                tcr_attn, pmhc_attn = out['attn']
                tcr_attn = tcr_attn.squeeze().detach().cpu().numpy()
                pmhc_attn = pmhc_attn.squeeze().detach().cpu().numpy()
            
            for i, (epi, mhc, cdr3a, cdr3b) in enumerate(zip(batch['epitope_seq'], batch['mhc_allele'], batch['cdr3_alpha_seq'], batch['cdr3_beta_seq'])):
                cdr3_embeddings['-'.join([cdr3a, cdr3b])] = tcr_projection[i, :]
                pmhc_embeddings['-'.join([epi, mhc])] = pmhc_projection[i, :]
                if self.model.agg == 'attn':
                    tcr_attn_dict['-'.join([cdr3a, cdr3b])] = tcr_attn[i, :]
                    pmhc_attn_dict['-'.join([epi, mhc])] = pmhc_attn[i, :]
        
        np.savez(os.path.join(self.log_dir, 'pmhc_embeddings.npz'), **pmhc_embeddings)
        np.savez(os.path.join(self.log_dir, 'cdr3_embeddings.npz'), **cdr3_embeddings)
        
        if self.model.agg == 'attn':
            np.savez(os.path.join(self.log_dir, 'pmhc_attn.npz'), **pmhc_attn_dict)
            np.savez(os.path.join(self.log_dir, 'cdr3_attn.npz'), **tcr_attn_dict)
    

class PairedCDR123pMHCCoembeddingTrainer(BaseTrainer):
    
    def __init__(self, config, log_dir=None):
        super(PairedCDR123pMHCCoembeddingTrainer, self).__init__(config, log_dir)
        
        self.model = PairedCDR123pMHCCoembeddingModel(**config.model)
        self.model.load_pretrained_pmhc_model(config.training.pretrained_pmhc_model)
        self.model.load_pretrained_tcr_model(config.training.pretrained_tcr_model)
        self.model.to(self.device)
        
        param_finetune_list = []
        param_default_list = []
        for name, param in self.model.named_parameters():
            if name.startswith('tcr_model'):
                param_finetune_list.append(param)
            elif name.startswith('pmhc_model.epitope_model'):
                param_finetune_list.append(param)
            elif name.startswith('pmhc_model.mhc_model'):
                param_finetune_list.append(param)
            else:
                param_default_list.append(param)
                
        self.optimizer, self.scheduler = self.configure_optimizer([
            {'params': param_finetune_list, 'lr': config.training.lr / 10},
            {'params': param_default_list, 'lr': config.training.lr},    
        ], config.training)
        
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        self.num_neg_pairs = config.training.non_binding_ratio # default: 5
        self.contrasruce_loss_type = config.training.contrasruce_loss_type
        self.contrastive_loss_coef = config.training.contrastive_loss_coef
        
        self.tcr_tokenizer = self.model.tokenizer
        self.tcr_feat_path = config.data.train_tcr_feat_path
        self.pmhc_bank = pd.read_table(config.data.train_pmhc_path)
        
        self.temperature = config.training.temperature
        self.margin = config.training.margin
        self.embed_hid_dim = config.model.embed_hid_dim
        
    def generate_nonbinding_tcr(self, batch):
        batch_size = len(batch['epitope_seq'])
        tcr_indices = []
        for epi, mhc in zip(batch['epitope_seq'], batch['mhc_allele']):
            pmhc_idx = self.pmhc_bank.loc[(self.pmhc_bank['Epitope.peptide'] == epi) & (self.pmhc_bank['MHC'] == mhc), 'pmhc_idx'].values[0]
            weights = (self.tcr_feat_bank['training_mask'][:, pmhc_idx] == 0).float() 
            sampled_indices = torch.multinomial(weights, self.num_neg_pairs, replacement=False)
            tcr_indices.append(sampled_indices)
            
        tcr_indices = torch.cat(tcr_indices, dim=0)
        batch_data = {
            'cdr1_alpha': batch['cdr1_alpha_seq'] + self.tcr_feat_bank['cdr1_alpha_seq'][tcr_indices].tolist(),
            'cdr1_beta': batch['cdr1_beta_seq'] + self.tcr_feat_bank['cdr1_beta_seq'][tcr_indices].tolist(),
            'cdr2_alpha':  batch['cdr2_alpha_seq'] + self.tcr_feat_bank['cdr2_alpha_seq'][tcr_indices].tolist(),
            'cdr2_beta':  batch['cdr2_beta_seq'] + self.tcr_feat_bank['cdr2_beta_seq'][tcr_indices].tolist(),
            'cdr3_alpha':  batch['cdr3_alpha_seq'] + self.tcr_feat_bank['cdr3_alpha_seq'][tcr_indices].tolist(),
            'cdr3_beta':  batch['cdr3_beta_seq'] + self.tcr_feat_bank['cdr3_beta_seq'][tcr_indices].tolist()
        }
        batch['tcr_token'], batch['segment_mask'] = paired_cdr123_batch_encoding(batch_data, self.tcr_tokenizer, batch_size * 6, max_cdr3_len=25)
        
        raw_index = np.arange(batch_size)
        repeated_index = list(raw_index) + list(np.repeat(raw_index, 5))
        batch['epitope_token'] = batch['epitope_token'][repeated_index, :]
        batch['mhc_embedding'] = batch['mhc_embedding'][repeated_index, :, :]
        batch['label'] = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 5)])

        return batch

    @staticmethod
    def compute_contrastive_loss(anchor, positive, negative, temperature=1.0):
        """
            anchor: B, E; positive B, E; negative: 5B, E
        """
        logits_pos = F.cosine_similarity(anchor, positive) # B
        negative = negative.view(-1, 5, anchor.size(1)) # B, 5, E
        logits_neg = torch.stack([F.cosine_similarity(anchor, neg) for neg in negative.transpose(0, 1)], dim=1) # B, 5
        
        numerator = torch.exp(logits_pos / temperature)
        denominator = numerator + torch.exp(logits_neg / temperature).sum(1)
        
        loss = - torch.log(numerator / denominator)
        return loss.mean()
    
    @staticmethod
    def compute_triplet_loss(anchor, positive, negative, margin=0.3):
        """
            anchor: B, E; positive B, E; negative: 5B, E
        """
        dist_pos = 1 - F.cosine_similarity(anchor, positive) # B
        negative = negative.view(-1, 5, anchor.size(1)) # B, 5, E
        dist_neg = torch.stack([1 - F.cosine_similarity(anchor, neg) for neg in negative.transpose(0, 1)], dim=1) # B, 5
        
        loss = F.relu(dist_pos.unsqueeze(1) - dist_neg + margin)
        return loss.mean()

    @staticmethod
    def compute_clf_metrics(y, pred, epitopes):
        avg_acc = accuracy_score(y, (pred >= 0.5).astype(int))
        avg_f1 = f1_score(y, (pred >= 0.5).astype(int))
        avg_auroc = roc_auc_score(y, pred)
        avg_auprc = average_precision_score(y, pred)
        
        res = pd.DataFrame({'epitope': epitopes, 'y': y, 'pred': pred})
        avg_epi_auroc = res.groupby('epitope').apply(lambda x: roc_auc_score(x['y'], x['pred'])).mean()
        avg_epi_auprc = res.groupby('epitope').apply(lambda x: average_precision_score(x['y'], x['pred'])).mean()
        
        return {
            'acc': avg_acc,
            'f1': avg_f1,
            'auc': avg_auroc,
            'aupr': avg_auprc,
            'auc_epi': avg_epi_auroc,
            'aupr_epi': avg_epi_auprc
        }
    
    def proceed_one_epoch(self, iterator, training=True, return_predictions=False):  
        avg_loss, data_size = 0, 0            
        pred, y, epitope_seqs = [], [], []
        
        if training:
            self.model.train()
            avg_bce_loss, avg_contrastive_loss = 0, 0
        else:
            self.model.eval()
            
        for batch in tqdm(iterator):
            if training:
                processed_batch = self.generate_nonbinding_tcr(batch) # sample nonbinding tcrs from the feature bank
            
            for key, item in batch.items():
                if isinstance(item, torch.Tensor):
                    batch[key] = item.to(self.device)
                    
            if training:
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(processed_batch)
                cdr_embed, cdr_padding_mask = self.model.compute_tcr_feat(processed_batch['tcr_token'], processed_batch['segment_mask'])
                out = self.model.forward(
                    epi_feat =epi_embed,
                    tcr_feat = cdr_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr_padding_mask
                )
                logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                tcr_projections, pmhc_projections = out['projection']
                labels = processed_batch['label']
                
                batch_size = labels.shape[0] // 6
                anchor_projections = pmhc_projections[:batch_size, :]
                positive_projections = tcr_projections[:batch_size, :]
                negative_projections = tcr_projections[batch_size:, :]
                
                bce_loss = F.binary_cross_entropy(logits, labels)
                if self.contrasruce_loss_type == 'simclr':
                    contrastive_loss = self.compute_contrastive_loss(anchor_projections, positive_projections, negative_projections, temperature=self.temperature)
                elif self.contrasruce_loss_type == 'triplet':
                    contrastive_loss = self.compute_triplet_loss(anchor_projections, positive_projections, negative_projections, margin=self.margin)
                elif self.contrasruce_loss_type == 'simclr+triplet':
                    contrastive_loss = self.compute_contrastive_loss(anchor_projections, positive_projections, negative_projections, temperature=self.temperature) + self.compute_triplet_loss(anchor_projections, positive_projections, negative_projections, margin=self.margin)
                    
                loss = bce_loss + self.contrastive_loss_coef * contrastive_loss
                epitope_seqs.extend(batch['epitope_seq'] + [seq for seq in batch['epitope_seq'] for _ in range(self.num_neg_pairs)])
                
            else:
                with torch.no_grad():
                    for key, item in batch.items():
                        if isinstance(item, torch.Tensor):
                            batch[key] = item.to(self.device)
                                 
                    epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                    cdr_embed, cdr_padding_mask = self.model.compute_tcr_feat(batch['tcr_token'], batch['segment_mask'])
                    
                    out = self.model.forward(
                        epi_feat = epi_embed,
                        tcr_feat = cdr_embed,
                        epi_padding_mask = epi_padding_mask,
                        tcr_padding_mask = cdr_padding_mask
                    )
                    logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                    labels = batch['label']
                    loss = F.binary_cross_entropy(logits, labels)
                    epitope_seqs.extend(batch['epitope_seq'])
                 
            pred.append(logits.detach().cpu().numpy())
            y.append(labels.detach().cpu().numpy())
            
            avg_loss += loss.item() * len(batch)
            data_size += len(batch)
            
            if training:
                avg_bce_loss += bce_loss.item() * len(batch)
                avg_contrastive_loss += contrastive_loss.item() * len(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)
        
        metrics = self.compute_clf_metrics(y, pred, epitope_seqs)
        metrics['loss'] = avg_loss / data_size
        metrics['bce_loss'] = avg_bce_loss / data_size if training else avg_loss / data_size
        metrics['contrastive_loss'] = avg_contrastive_loss / data_size if training else 0.0
        
        if return_predictions:
            return metrics, y, pred
        else:
            return metrics
    
    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)
    
    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        return self.proceed_one_epoch(eval_loader, training=False, return_predictions=return_predictions)
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)
        self.tcr_feat_bank = load_paired_cdr123_feat_bank(self.tcr_feat_path, num_pmhc=901)
        self.tcr_feat_bank['training_mask'] = self.tcr_feat_bank['binding_mask'].clone()
        
        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            
            self.logger.info(f'Train Loss: {train_metrics["loss"]:.3f} | BCE Loss: {train_metrics["bce_loss"]:.3f} | Contrastive Loss: {train_metrics["contrastive_loss"]:.3f}')
            self.logger.info(f'Train ACC: {train_metrics["acc"]*100:.2f}% | F1: {train_metrics["f1"]:.3f} | AUROC {train_metrics["auc"]:.3f} | AUPRC: {train_metrics["aupr"]:.3f}')
            self.logger.info(f'Train AVG AUROC: {train_metrics["auc_epi"]:.3f} | AVG AUPRC: {train_metrics["aupr_epi"]:.3f}')
            
            self.logger.info(f'Valid Loss: {val_metrics["loss"]:.3f} | BCE Loss: {val_metrics["bce_loss"]:.3f} | Contrastive Loss: {val_metrics["contrastive_loss"]:.3f}')
            self.logger.info(f'Valid ACC: {val_metrics["acc"]*100:.2f}% | F1: {val_metrics["f1"]:.3f} | AUROC {val_metrics["auc"]:.3f} | AUPRC: {val_metrics["aupr"]:.3f}')
            self.logger.info(f'Valid AVG AUROC: {val_metrics["auc_epi"]:.3f} | AVG AUPRC: {val_metrics["aupr_epi"]:.3f}')
            
            for key, value in train_metrics.items(): 
                self.writer.add_scalar(f'Train/{key}', value, epoch+1)
            for key, value in val_metrics.items(): 
                self.writer.add_scalar(f'Valid/{key}', value, epoch+1)

            if epoch >= 19:
                # self.early_stopping(val_metrics["aupr"], self.model, goal="maximize")
                self.early_stopping(val_metrics["auc"], self.model, goal="maximize")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break

        self.writer.close()
        
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        test_metrics, y, pred = self.evaluate_one_epoch(test_loader, return_predictions=True)
        self.logger.info(f'Test Loss: {test_metrics["loss"]:.3f} | BCE Loss: {test_metrics["bce_loss"]:.3f} | Contrastive Loss: {test_metrics["contrastive_loss"]:.3f}')
        self.logger.info(f'Test ACC: {test_metrics["acc"]*100:.2f}% | F1: {test_metrics["f1"]:.3f} | AUROC {test_metrics["auc"]:.3f} | AUPRC: {test_metrics["aupr"]:.3f}')
        self.logger.info(f'Test AVG AUROC: {test_metrics["auc_epi"]:.3f} | AVG AUPRC: {test_metrics["aupr_epi"]:.3f}')
        for key, value in test_metrics.items(): 
            self.writer.add_scalar(f'Test/{key}', value)
        
        res = pd.DataFrame({'Target': y, 'Pred': pred})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)
        
    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, map_location='cpu'), strict=False)
        preds, sims, epi, mhc, cdr1_alpha, cdr1_beta, cdr2_alpha, cdr2_beta, cdr3_alpha, cdr3_beta = [], [], [], [], [], [], [], [], [], []
            
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                                 
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                cdr_embed, cdr_padding_mask = self.model.compute_tcr_feat(batch['tcr_token'], batch['segment_mask'])
                
                out = self.model.forward(
                    epi_feat = epi_embed,
                    tcr_feat = cdr_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr_padding_mask
                )
                logits, similarity = out['logits'].squeeze(-1), out['dist'].squeeze(-1)
                preds.append(logits.detach().cpu().numpy())
                sims.append(similarity.detach().cpu().numpy())
                epi.extend(batch['epitope_seq'])
                mhc.extend(batch['mhc_allele'])
                cdr1_alpha.extend(batch['cdr1_alpha_seq'])
                cdr1_beta.extend(batch['cdr1_beta_seq'])
                cdr2_alpha.extend(batch['cdr2_alpha_seq'])
                cdr2_beta.extend(batch['cdr2_beta_seq'])
                cdr3_alpha.extend(batch['cdr3_alpha_seq'])
                cdr3_beta.extend(batch['cdr3_beta_seq'])
                
            preds = np.concatenate(preds)
            sims = np.concatenate(sims)
            df = pd.DataFrame({'Epitope.peptide': epi, 'MHC': mhc, 'CDR1.alpha.aa': cdr1_alpha, 'CDR1.beta.aa': cdr1_beta,
                               'CDR2.alpha.aa': cdr2_alpha, 'CDR2.beta.aa': cdr2_beta, 'CDR3.alpha.aa': cdr3_alpha, 'CDR3.beta.aa': cdr3_beta,
                               'Pred': preds, 'Similarity': sims})
            df.to_csv(os.path.join(self.log_dir, 'predictions.csv'), index=False)   
            
    def predict_embeddings(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location), strict=False)
        pmhc_embeddings, cdr3_embeddings = {}, {}
        if self.model.agg == 'attn':
            pmhc_attn_dict, tcr_attn_dict = {}, {}
            
        self.model.eval()
        for batch in tqdm(data_loader):
            for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                                
            epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
            cdr_embed, cdr_padding_mask = self.model.compute_tcr_feat(batch['tcr_token'], batch['segment_mask'])
            
            out = self.model.forward(
                epi_feat = epi_embed,
                tcr_feat = cdr_embed,
                epi_padding_mask = epi_padding_mask,
                tcr_padding_mask = cdr_padding_mask
            )
            tcr_projection, pmhc_projection = out['projection']
            tcr_projection = tcr_projection.detach().cpu().numpy()
            pmhc_projection = pmhc_projection.detach().cpu().numpy()
            
            if self.model.agg == 'attn':
                tcr_attn, pmhc_attn = out['attn']
                tcr_attn = tcr_attn.squeeze().detach().cpu().numpy()
                pmhc_attn = pmhc_attn.squeeze().detach().cpu().numpy()
            
            for i, (epi, mhc, cdr1a, cdr1b, cdr2a, cdr2b, cdr3a, cdr3b) in enumerate(zip(
                batch['epitope_seq'], batch['mhc_allele'], batch['cdr1_alpha_seq'], batch['cdr1_beta_seq'], batch['cdr2_alpha_seq'], batch['cdr2_beta_seq'], batch['cdr3_alpha_seq'], batch['cdr3_beta_seq'])):
                cdr3_embeddings['-'.join([cdr1a, cdr2a, cdr3a, cdr1b, cdr2b, cdr3b])] = tcr_projection[i, :]
                pmhc_embeddings['-'.join([epi, mhc])] = pmhc_projection[i, :]
                if self.model.agg == 'attn':
                    tcr_attn_dict['-'.join([cdr1a, cdr2a, cdr3a, cdr1b, cdr2b, cdr3b])] = tcr_attn[i, :]
                    pmhc_attn_dict['-'.join([epi, mhc])] = pmhc_attn[i, :]
        
        np.savez(os.path.join(self.log_dir, 'pmhc_embeddings.npz'), **pmhc_embeddings)
        np.savez(os.path.join(self.log_dir, 'cdr_embeddings.npz'), **cdr3_embeddings)
        
        if self.model.agg == 'attn':
            np.savez(os.path.join(self.log_dir, 'pmhc_attn.npz'), **pmhc_attn_dict)
            np.savez(os.path.join(self.log_dir, 'cdr_attn.npz'), **tcr_attn_dict)
