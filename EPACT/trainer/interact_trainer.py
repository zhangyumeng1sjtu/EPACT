import time
import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseTrainer
from ..model import PairedCDR3pMHCCoembeddingModel, PairedCDR123pMHCCoembeddingModel
from ..model.utils import EarlyStopping, unfreeze_params
from ..utils.misc import get_scores_dist, get_scores_contact


class PairedCDR3pMHCInteractTrainer(BaseTrainer):
    def __init__(self, config, log_dir=None):
        super(PairedCDR3pMHCInteractTrainer, self).__init__(config, log_dir)    

        self.model = PairedCDR3pMHCCoembeddingModel(**config.model)
        
        self.model.load_pretrained_tcr_model(config.training.pretrained_tcr_model)
        self.model.load_pretrained_pmhc_model(config.training.pretrained_pmhc_model)
        if config.training.pretrained_tcr_pmhc_model is not None:
            self.model.load_state_dict(torch.load(config.training.pretrained_tcr_pmhc_model, map_location='cpu'), strict=False)
        
        if config.training.finetune:
            unfreeze_params([self.model])
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
        
        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        self.alpha_coef = config.training.alpha_coef
        self.contact_coef = config.training.contact_coef
    
    @staticmethod
    def get_loss(pred, labels, chain='beta'):
        dist = labels['dist']
        contact = labels['contact']
        mask = labels['alpha_masking'] if chain == 'alpha' else labels['beta_masking']
        dist_loss = F.mse_loss(pred[..., 0], dist, reduction='none')
        dist_loss = dist_loss * mask / (dist + 0.01)
        dist_loss = torch.sum(dist_loss) / torch.sum(mask)
        
        contact_loss = F.binary_cross_entropy(pred[..., 1], contact, reduction='none')
        contact_loss = contact_loss * mask
        contact_loss = torch.sum(contact_loss) / torch.sum(mask)
        
        return dist_loss, contact_loss
    
    @staticmethod  
    def get_scores(pred, labels):
        dist, contact, mask = labels
        avg_metrics_dist, metrics_dist = get_scores_dist(dist, pred[0], mask)
        avg_metrics_bd, metrics_bd = get_scores_contact(contact, pred[1], mask)
        return avg_metrics_dist + avg_metrics_bd, metrics_dist + metrics_bd
    
    def proceed_one_epoch(self, iterator, training=True, return_predictions=False):
        avg_alpha_dist_loss = 0
        avg_alpha_contact_loss = 0
        avg_beta_dist_loss = 0
        avg_beta_contact_loss = 0
        data_size = 0
        
        pred_dist, pred_contact, dist, contact, alpha_mask, beta_mask = [], [], [], [], [], []
        
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        for batch in tqdm(iterator):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            for key, value in batch['target'].items():
                if isinstance(value, torch.Tensor):
                    batch['target'][key] = value.to(self.device)

            epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
            cdr3_embed, cdr3_padding_mask, cdr3_sep_mask = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
                    
            out = self.model.forward(
                epi_feat = epi_embed,
                tcr_feat = cdr3_embed,
                epi_padding_mask = epi_padding_mask,
                tcr_padding_mask = cdr3_padding_mask,
                tcr_sep_mask = cdr3_sep_mask,
                predict_dist_map = True
            )
            targets = batch['target']
            alpha_dist_loss, alpha_contact_loss = self.get_loss(out, targets, chain='alpha')
            beta_dist_loss, beta_contact_loss = self.get_loss(out, targets, chain='beta')
            
            pred_dist.extend(out[:, :, :, 0].detach().cpu().numpy().tolist())
            pred_contact.extend(out[:, :, :, 1].detach().cpu().numpy().tolist())
            
            dist.extend(targets['dist'].detach().cpu().numpy().tolist())
            contact.extend(targets['contact'].detach().cpu().numpy().tolist())
            alpha_mask.extend(targets['alpha_masking'].detach().cpu().numpy().tolist())
            beta_mask.extend(targets['beta_masking'].detach().cpu().numpy().tolist())

            avg_alpha_dist_loss += alpha_dist_loss.cpu().item() * len(targets)
            avg_alpha_contact_loss += alpha_contact_loss.cpu().item() * len(targets)
            avg_beta_dist_loss += beta_dist_loss.cpu().item() * len(targets)
            avg_beta_contact_loss += beta_contact_loss.cpu().item() * len(targets)
            data_size += len(targets)
            
            if training:
                loss = self.alpha_coef * (alpha_dist_loss + self.contact_coef * alpha_contact_loss) + beta_dist_loss + self.contact_coef * beta_contact_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        avg_alpha_dist_loss = avg_alpha_dist_loss / data_size
        avg_alpha_contact_loss = avg_alpha_contact_loss / data_size
        avg_beta_dist_loss = avg_beta_dist_loss / data_size
        avg_beta_contact_loss = avg_beta_contact_loss / data_size
        
        alpha_scores, alpha_scores_samples = self.get_scores([pred_dist, pred_contact], [dist, contact, alpha_mask])
        beta_scores, beta_scores_samples = self.get_scores([pred_dist, pred_contact], [dist, contact, beta_mask])
        
        alpha_metrics = {
            'loss': avg_alpha_dist_loss +  self.contact_coef * avg_alpha_contact_loss,
            'dist': avg_alpha_dist_loss,
            'contact': avg_alpha_contact_loss,
            'corr': alpha_scores[0],
            'mae': alpha_scores[1],
            'mape': alpha_scores[2],
            'auc': alpha_scores[3],
            'scores': alpha_scores_samples
        }
        beta_metrics = {
            'loss': avg_beta_dist_loss +  self.contact_coef * avg_beta_contact_loss,
            'dist': avg_beta_dist_loss,
            'contact': avg_beta_contact_loss,
            'corr': beta_scores[0],
            'mae': beta_scores[1],
            'mape': beta_scores[2],
            'auc': beta_scores[3],
            'scores': beta_scores_samples
        }
        
        if return_predictions:
            preds = {'dist': pred_dist, 'contact': pred_contact, 'alpha_mask': alpha_mask, 'beta_mask': beta_mask}
            return alpha_metrics, beta_metrics, preds
        else:
            return alpha_metrics, beta_metrics
        
    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)

    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        return self.proceed_one_epoch(eval_loader, training=False, return_predictions=return_predictions)
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)
         
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            train_alpha_metrics, train_beta_metrics = self.train_one_epoch(train_loader)
            val_alpha_metrics, val_beta_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()
            
            end_time = time.time()
            epoch_secs = end_time - start_time
            
            train_loss = self.alpha_coef * train_alpha_metrics["loss"] +  train_beta_metrics["loss"]
            train_dist_loss = self.alpha_coef * train_alpha_metrics["dist"] + train_beta_metrics["dist"]
            train_contact_loss = self.alpha_coef * train_alpha_metrics["contact"] + train_beta_metrics["contact"]
            val_loss = self.alpha_coef * val_alpha_metrics["loss"] +  val_beta_metrics["loss"]
            val_dist_loss = self.alpha_coef * val_alpha_metrics["dist"] + val_beta_metrics["dist"]
            val_contact_loss = self.alpha_coef * val_alpha_metrics["contact"] + val_beta_metrics["contact"]
            
            self.logger.info( f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.3f} | Dist Loss: {train_dist_loss:.3f} | Contact Loss: {train_contact_loss:.3f}')
            self.logger.info(f'Train CDR3 Alpha PCC: {train_alpha_metrics["corr"]:.3f} | MAE: {train_alpha_metrics["mae"]:.3f} | MAPE: {train_alpha_metrics["mape"]:.3f} | AUC: {train_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Train CDR3 Beta PCC: {train_beta_metrics["corr"]:.3f} | MAE: {train_beta_metrics["mae"]:.3f} | MAPE: {train_beta_metrics["mape"]:.3f} | AUC: {train_beta_metrics["auc"]:.3f}')
            
            self.logger.info(f'Valid Loss: {val_loss:.3f} | Dist Loss: {val_dist_loss:.3f} | Contact Loss: {val_contact_loss:.3f}')
            self.logger.info(f'Valid CDR3 Alpha PCC: {val_alpha_metrics["corr"]:.3f} | MAE: {val_alpha_metrics["mae"]:.3f} | MAPE: {val_alpha_metrics["mape"]:.3f} | AUC: {val_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Valid CDR3 Beta PCC: {val_beta_metrics["corr"]:.3f} | MAE: {val_beta_metrics["mae"]:.3f} | MAPE: {val_beta_metrics["mape"]:.3f} | AUC: {val_beta_metrics["auc"]:.3f}')
            
            for key, value in train_alpha_metrics.items():
                if key != 'scores':
                    self.writer.add_scalar(f'Train/alpha/{key}', value, epoch+1)
            for key, value in train_beta_metrics.items():
                if key != 'scores':
                    self.writer.add_scalar(f'Train/beta/{key}', value, epoch+1)
            for key, value in val_alpha_metrics.items(): 
                if key != 'scores':
                    self.writer.add_scalar(f'Valid/alpha/{key}', value, epoch+1)
            for key, value in val_beta_metrics.items(): 
                if key != 'scores':
                    self.writer.add_scalar(f'Valid/beta/{key}', value, epoch+1)
            
            self.early_stopping(val_loss, self.model, goal="minimize")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
            
        self.writer.close()
        
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        test_alpha_metrics, test_beta_metrics, test_preds = self.evaluate_one_epoch(test_loader, return_predictions=True)
        
        test_dist_loss = self.alpha_coef * test_alpha_metrics['dist'] + test_beta_metrics['dist']
        test_contact_loss = self.alpha_coef * test_alpha_metrics['contact'] + test_beta_metrics['contact']
        test_total_loss = self.alpha_coef * test_alpha_metrics['loss'] + test_beta_metrics['loss']
        
        self.logger.info( f'Test Loss: {test_total_loss:.3f} | Dist Loss: {test_dist_loss:.3f} | Contact Loss: {test_contact_loss:.3f}')
        self.logger.info(f'Test CDR3 Alpha PCC: {test_alpha_metrics["corr"]:.3f} | MAE: {test_alpha_metrics["mae"]:.3f} | MAPE: {test_alpha_metrics["mape"]:.3f} | AUC: {test_alpha_metrics["auc"]:.3f}')
        self.logger.info(f'Test CDR3 Beta PCC: {test_beta_metrics["corr"]:.3f} | MAE: {test_beta_metrics["mae"]:.3f} | MAPE: {test_beta_metrics["mape"]:.3f} | AUC: {test_beta_metrics["auc"]:.3f}')
        
        with open(os.path.join(self.log_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump({'alpha': test_alpha_metrics['scores'], 'beta': test_beta_metrics['scores']}, f)
            
        with open(os.path.join(self.log_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump(test_preds, f)
        
    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        pred_dist, pred_contact, epi, mhc, cdr3_alpha, cdr3_beta = [], [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                                 
                epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
                cdr3_embed, cdr3_padding_mask, cdr3_sep_mask = self.model.compute_tcr_feat(batch['cdr3_token'], batch['chain_mask'])
                
                out = self.model.forward(
                    epi_feat = epi_embed,
                    tcr_feat = cdr3_embed,
                    epi_padding_mask = epi_padding_mask,
                    tcr_padding_mask = cdr3_padding_mask,
                    tcr_sep_mask = cdr3_sep_mask,
                    predict_dist_map = True
                )
                pred_dist.extend(out[:, :, :, 0].detach().cpu().numpy().tolist())
                pred_contact.extend(out[:, :, :, 1].detach().cpu().numpy().tolist())
                epi.extend(batch['epitope_seq'])
                mhc.extend(batch['mhc_allele'])
                cdr3_alpha.extend(batch['cdr3_alpha_seq'])
                cdr3_beta.extend(batch['cdr3_beta_seq'])
            
            res = [{'dist': dist, 'contact': contact, 'epitope': epi_, 'mhc': mhc_, 'cdr3.alpha': cdr3_alpha_, 'cdr3.beta': cdr3_beta_} for (dist, contact, epi_, mhc_, cdr3_alpha_, cdr3_beta_) in zip(pred_dist, pred_contact, epi, mhc, cdr3_alpha, cdr3_beta)]
            
            with open(os.path.join(self.log_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump(res, f)


class PairedCDR123pMHCInteractTrainer(BaseTrainer):
    def __init__(self, config, log_dir=None):
        super(PairedCDR123pMHCInteractTrainer, self).__init__(config, log_dir)    

        self.model = PairedCDR123pMHCCoembeddingModel(**config.model)
        
        self.model.load_pretrained_tcr_model(config.training.pretrained_tcr_model)
        self.model.load_pretrained_pmhc_model(config.training.pretrained_pmhc_model)
        if config.training.pretrained_tcr_pmhc_model is not None:
            self.model.load_state_dict(torch.load(config.training.pretrained_tcr_pmhc_model, map_location='cpu'), strict=False)
        
        if config.training.finetune:
            unfreeze_params([self.model])
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
        
        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        self.cdr1a_coef = config.training.cdr1a_coef
        self.cdr3a_coef = config.training.cdr3a_coef
        self.contact_coef = config.training.contact_coef
        
    @staticmethod
    def get_loss(pred, labels, mask):
        
        dist = labels['dist']
        contact = labels['contact']

        dist_loss = F.mse_loss(pred[..., 0], dist, reduction='none')
        dist_loss = dist_loss * mask / (dist + 0.01)
        dist_loss = torch.sum(dist_loss) / torch.sum(mask)
        
        contact_loss = F.binary_cross_entropy(pred[..., 1], contact, reduction='none')
        contact_loss = contact_loss * mask
        contact_loss = torch.sum(contact_loss) / torch.sum(mask)
        
        return dist_loss, contact_loss
    
    @staticmethod  
    def get_scores(pred, labels):
        dist, contact, mask = labels
        avg_metrics_dist, metrics_dist = get_scores_dist(dist, pred[0], mask)
        avg_metrics_bd, metrics_bd = get_scores_contact(contact, pred[1], mask)
        return avg_metrics_dist + avg_metrics_bd, metrics_dist + metrics_bd
    
    def proceed_one_epoch(self, iterator, training=True, return_predictions=False):
        avg_cdr1_alpha_dist_loss = 0
        avg_cdr1_alpha_contact_loss = 0
        avg_cdr3_alpha_dist_loss = 0
        avg_cdr3_alpha_contact_loss = 0
        avg_cdr3_beta_dist_loss = 0
        avg_cdr3_beta_contact_loss = 0
        data_size = 0
        
        pred_dist, pred_contact, dist, contact, cdr1_alpha_mask, cdr3_alpha_mask, cdr3_beta_mask = [], [], [], [], [], [], []
        
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        for batch in tqdm(iterator):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            for key, value in batch[f'target'].items():
                if isinstance(value, torch.Tensor):
                    batch['target'][key] = value.to(self.device)
                            
            epi_embed, epi_padding_mask = self.model.compute_pmhc_feat(batch)
            cdr_embed, cdr_padding_mask = self.model.compute_tcr_feat(batch['tcr_token'], batch['segment_mask'])
            
            out = self.model.forward(
                epi_feat = epi_embed,
                tcr_feat = cdr_embed,
                epi_padding_mask = epi_padding_mask,
                tcr_padding_mask = cdr_padding_mask,
                predict_dist_map = True,
                tcr_segment_mask = batch['segment_mask']
            )
            
            segment_mask = batch['segment_mask'][:, 1:]
            epi_mask = ~epi_padding_mask[:, 1:]
            targets = batch['target']
            cdr1_alpha_mask_ = torch.logical_and(segment_mask.eq(1).unsqueeze(2), epi_mask.unsqueeze(1))
            cdr3_alpha_mask_ = torch.logical_and(segment_mask.eq(3).unsqueeze(2), epi_mask.unsqueeze(1))
            cdr3_beta_mask_ = torch.logical_and(segment_mask.eq(6).unsqueeze(2), epi_mask.unsqueeze(1))
            
            cdr1_alpha_dist_loss, cdr1_alpha_contact_loss = self.get_loss(out, targets, mask=cdr1_alpha_mask_)
            cdr3_alpha_dist_loss, cdr3_alpha_contact_loss = self.get_loss(out, targets, mask=cdr3_alpha_mask_)
            cdr3_beta_dist_loss, cdr3_beta_contact_loss = self.get_loss(out, targets, mask=cdr3_beta_mask_)
            
            pred_dist.extend(out[:, :, :, 0].detach().cpu().numpy().tolist())
            pred_contact.extend(out[:, :, :, 1].detach().cpu().numpy().tolist())
            
            dist.extend(targets['dist'].detach().cpu().numpy().tolist())
            contact.extend(targets['contact'].detach().cpu().numpy().tolist())
            cdr1_alpha_mask.extend(cdr1_alpha_mask_.detach().cpu().numpy().tolist())
            cdr3_alpha_mask.extend(cdr3_alpha_mask_.detach().cpu().numpy().tolist())
            cdr3_beta_mask.extend(cdr3_beta_mask_.detach().cpu().numpy().tolist())

            avg_cdr1_alpha_dist_loss += cdr1_alpha_dist_loss.cpu().item() * len(targets)
            avg_cdr1_alpha_contact_loss += cdr1_alpha_contact_loss.cpu().item() * len(targets)
            avg_cdr3_alpha_dist_loss += cdr3_alpha_dist_loss.cpu().item() * len(targets)
            avg_cdr3_alpha_contact_loss += cdr3_alpha_contact_loss.cpu().item() * len(targets)
            avg_cdr3_beta_dist_loss += cdr3_beta_dist_loss.cpu().item() * len(targets)
            avg_cdr3_beta_contact_loss += cdr3_beta_contact_loss.cpu().item() * len(targets)
            data_size += len(targets)
            
            if training:
                loss = self.cdr1a_coef * (cdr1_alpha_dist_loss + self.contact_coef * cdr1_alpha_contact_loss) + self.cdr3a_coef * (cdr3_alpha_dist_loss + self.contact_coef * cdr3_alpha_contact_loss) + cdr3_beta_dist_loss + self.contact_coef * cdr3_beta_contact_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        avg_cdr1_alpha_dist_loss = avg_cdr1_alpha_dist_loss / data_size
        avg_cdr1_alpha_contact_loss = avg_cdr1_alpha_contact_loss / data_size
        avg_cdr3_alpha_dist_loss = avg_cdr3_alpha_dist_loss / data_size
        avg_cdr3_alpha_contact_loss = avg_cdr3_alpha_contact_loss / data_size
        avg_cdr3_beta_dist_loss = avg_cdr3_beta_dist_loss / data_size
        avg_cdr3_beta_contact_loss = avg_cdr3_beta_contact_loss / data_size
        
        cdr1_alpha_scores, cdr1_alpha_scores_samples = self.get_scores([pred_dist, pred_contact], [dist, contact, cdr1_alpha_mask])
        cdr3_alpha_scores, cdr3_alpha_scores_samples = self.get_scores([pred_dist, pred_contact], [dist, contact, cdr3_alpha_mask])
        cdr3_beta_scores, cdr3_beta_scores_samples = self.get_scores([pred_dist, pred_contact], [dist, contact, cdr3_beta_mask])        
        
        cdr1_alpha_metrics = {
            'loss': avg_cdr1_alpha_dist_loss +  self.contact_coef * avg_cdr1_alpha_contact_loss,
            'dist': avg_cdr1_alpha_dist_loss,
            'contact': avg_cdr1_alpha_contact_loss,
            'corr': cdr1_alpha_scores[0],
            'mae': cdr1_alpha_scores[1],
            'mape': cdr1_alpha_scores[2],
            'auc': cdr1_alpha_scores[3],
            'scores': cdr1_alpha_scores_samples
        }
        cdr3_alpha_metrics = {
            'loss': avg_cdr3_alpha_dist_loss +  self.contact_coef * avg_cdr3_alpha_contact_loss,
            'dist': avg_cdr3_alpha_dist_loss,
            'contact': avg_cdr3_alpha_contact_loss,
            'corr': cdr3_alpha_scores[0],
            'mae': cdr3_alpha_scores[1],
            'mape': cdr3_alpha_scores[2],
            'auc': cdr3_alpha_scores[3],
            'scores': cdr3_alpha_scores_samples
        }
        cdr3_beta_metrics = {
            'loss': avg_cdr3_beta_dist_loss +  self.contact_coef * avg_cdr3_beta_contact_loss,
            'dist': avg_cdr3_beta_dist_loss,
            'contact': avg_cdr3_beta_contact_loss,
            'corr': cdr3_beta_scores[0],
            'mae': cdr3_beta_scores[1],
            'mape': cdr3_beta_scores[2],
            'auc': cdr3_beta_scores[3],
            'scores': cdr3_beta_scores_samples
        }
        
        if return_predictions:
            preds = {'dist': pred_dist, 'contact': pred_contact, 'cdr1_alpha_mask': cdr1_alpha_mask, 'cdr3_alpha_mask': cdr3_alpha_mask, 'cdr3_beta_mask':cdr3_beta_mask}
            return cdr1_alpha_metrics, cdr3_alpha_metrics, cdr3_beta_metrics, preds
        else:
            return cdr1_alpha_metrics, cdr3_alpha_metrics, cdr3_beta_metrics
        
    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)

    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        return self.proceed_one_epoch(eval_loader, training=False, return_predictions=return_predictions)
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)
         
        for epoch in range(self.max_epochs):
            start_time = time.time()
            
            train_cdr1_alpha_metrics, train_cdr3_alpha_metrics, train_cdr3_beta_metrics = self.train_one_epoch(train_loader)
            val_cdr1_alpha_metrics, val_cdr3_alpha_metrics, val_cdr3_beta_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()
            
            end_time = time.time()
            epoch_secs = end_time - start_time
            
            train_loss = self.cdr1a_coef * train_cdr1_alpha_metrics["loss"] + self.cdr3a_coef * train_cdr3_alpha_metrics["loss"] + train_cdr3_beta_metrics["loss"]
            train_dist_loss = self.cdr1a_coef * train_cdr1_alpha_metrics["dist"] + self.cdr3a_coef * train_cdr3_alpha_metrics["dist"] + train_cdr3_beta_metrics["dist"]
            train_contact_loss = self.cdr1a_coef * train_cdr1_alpha_metrics["contact"] + self.cdr3a_coef * train_cdr3_alpha_metrics["contact"] + train_cdr3_beta_metrics["contact"]
            val_loss = self.cdr1a_coef * val_cdr1_alpha_metrics["loss"] + self.cdr3a_coef * val_cdr3_alpha_metrics["loss"] + val_cdr3_beta_metrics["loss"]
            val_dist_loss = self.cdr1a_coef * val_cdr1_alpha_metrics["dist"] +self.cdr3a_coef * val_cdr3_alpha_metrics["dist"] + val_cdr3_beta_metrics["dist"]
            val_contact_loss = self.cdr1a_coef * val_cdr1_alpha_metrics["contact"] + self.cdr3a_coef * val_cdr3_alpha_metrics["contact"] + val_cdr3_beta_metrics["contact"]
            
            self.logger.info( f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_loss:.3f} | Dist Loss: {train_dist_loss:.3f} | Contact Loss: {train_contact_loss:.3f}')
            self.logger.info(f'Train CDR1 Alpha PCC: {train_cdr1_alpha_metrics["corr"]:.3f} | MAE: {train_cdr1_alpha_metrics["mae"]:.3f} | MAPE: {train_cdr1_alpha_metrics["mape"]:.3f} | AUC: {train_cdr1_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Train CDR3 Alpha PCC: {train_cdr3_alpha_metrics["corr"]:.3f} | MAE: {train_cdr3_alpha_metrics["mae"]:.3f} | MAPE: {train_cdr3_alpha_metrics["mape"]:.3f} | AUC: {train_cdr3_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Train CDR3 Beta PCC: {train_cdr3_beta_metrics["corr"]:.3f} | MAE: {train_cdr3_beta_metrics["mae"]:.3f} | MAPE: {train_cdr3_beta_metrics["mape"]:.3f} | AUC: {train_cdr3_beta_metrics["auc"]:.3f}')
            
            self.logger.info(f'Valid Loss: {val_loss:.3f} | Dist Loss: {val_dist_loss:.3f} | Contact Loss: {val_contact_loss:.3f}')
            self.logger.info(f'Valid CDR1 Alpha PCC: {val_cdr1_alpha_metrics["corr"]:.3f} | MAE: {val_cdr1_alpha_metrics["mae"]:.3f} | MAPE: {val_cdr1_alpha_metrics["mape"]:.3f} | AUC: {val_cdr1_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Valid CDR3 Alpha PCC: {val_cdr3_alpha_metrics["corr"]:.3f} | MAE: {val_cdr3_alpha_metrics["mae"]:.3f} | MAPE: {val_cdr3_alpha_metrics["mape"]:.3f} | AUC: {val_cdr3_alpha_metrics["auc"]:.3f}')
            self.logger.info(f'Valid CDR3 Beta PCC: {val_cdr3_beta_metrics["corr"]:.3f} | MAE: {val_cdr3_beta_metrics["mae"]:.3f} | MAPE: {val_cdr3_beta_metrics["mape"]:.3f} | AUC: {val_cdr3_beta_metrics["auc"]:.3f}')
            
            for key, value in train_cdr1_alpha_metrics.items():
                if key != 'scores':
                    self.writer.add_scalar(f'Train/cdr1_alpha/{key}', value, epoch+1)
            for key, value in train_cdr3_alpha_metrics.items():
                if key != 'scores':
                    self.writer.add_scalar(f'Train/cdr3_alpha/{key}', value, epoch+1)
            for key, value in train_cdr3_beta_metrics.items():
                if key != 'scores':
                    self.writer.add_scalar(f'Train/cdr3_beta/{key}', value, epoch+1)
            for key, value in val_cdr1_alpha_metrics.items(): 
                if key != 'scores':
                    self.writer.add_scalar(f'Valid/cdr1_alpha/{key}', value, epoch+1)
            for key, value in val_cdr3_alpha_metrics.items(): 
                if key != 'scores':
                    self.writer.add_scalar(f'Valid/cdr3_alpha/{key}', value, epoch+1)
            for key, value in val_cdr3_beta_metrics.items(): 
                if key != 'scores':
                    self.writer.add_scalar(f'Valid/cdr3_beta/{key}', value, epoch+1)
            
            early_stopping_criterion = val_loss
            self.early_stopping(early_stopping_criterion, self.model, goal="minimize")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
            
        self.writer.close()
        
    def test(self, test_loader, model_location, out_dir=None):
        if out_dir is None:
            out_dir = self.log_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        self.model.load_state_dict(torch.load(model_location))
        test_cdr1_alpha_metrics, test_cdr3_alpha_metrics, test_cdr3_beta_metrics , test_preds = self.evaluate_one_epoch(test_loader, return_predictions=True)
        
        test_loss = self.cdr1a_coef * test_cdr1_alpha_metrics["loss"] + self.cdr3a_coef * test_cdr3_alpha_metrics["loss"] + test_cdr3_beta_metrics["loss"]
        test_dist_loss = self.cdr1a_coef * test_cdr1_alpha_metrics["dist"] + self.cdr3a_coef * test_cdr3_alpha_metrics["dist"] + test_cdr3_beta_metrics["dist"]
        test_contact_loss = self.cdr1a_coef * test_cdr1_alpha_metrics["contact"] + self.cdr3a_coef * test_cdr3_alpha_metrics["contact"] + test_cdr3_beta_metrics["contact"]
        
        self.logger.info(f'Test Loss: {test_loss:.3f} | Dist Loss: {test_dist_loss:.3f} | Contact Loss: {test_contact_loss:.3f}')
        self.logger.info(f'Test CDR1 Alpha PCC: {test_cdr1_alpha_metrics["corr"]:.3f} | MAE: {test_cdr1_alpha_metrics["mae"]:.3f} | MAPE: {test_cdr1_alpha_metrics["mape"]:.3f} | AUC: {test_cdr1_alpha_metrics["auc"]:.3f}')
        self.logger.info(f'Test CDR3 Alpha PCC: {test_cdr3_alpha_metrics["corr"]:.3f} | MAE: {test_cdr3_alpha_metrics["mae"]:.3f} | MAPE: {test_cdr3_alpha_metrics["mape"]:.3f} | AUC: {test_cdr3_alpha_metrics["auc"]:.3f}')
        self.logger.info(f'Test CDR3 Beta PCC: {test_cdr3_beta_metrics["corr"]:.3f} | MAE: {test_cdr3_beta_metrics["mae"]:.3f} | MAPE: {test_cdr3_beta_metrics["mape"]:.3f} | AUC: {test_cdr3_beta_metrics["auc"]:.3f}')
        
        with open(os.path.join(out_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump({'cdr1_alpha': test_cdr1_alpha_metrics['scores'], 'cdr3_alpha': test_cdr3_alpha_metrics['scores'], 'cdr3_beta': test_cdr3_beta_metrics['scores']}, f)
            
        with open(os.path.join(out_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump(test_preds, f)
        
    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        self.model.eval()
        
        pred_dist, pred_contact, cdr1_alpha_mask, cdr3_alpha_mask, cdr3_beta_mask = [], [], [], [], []
        epi, mhc, cdr1_alpha, cdr3_alpha, cdr3_beta = [], [], [], [], []
        
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
                    tcr_padding_mask = cdr_padding_mask,
                    predict_dist_map = True,
                    tcr_segment_mask = batch['segment_mask']
                )
                
                segment_mask = batch['segment_mask'][:, 1:]
                epi_mask = ~epi_padding_mask[:, 1:]
                cdr1_alpha_mask_ = torch.logical_and(segment_mask.eq(1).unsqueeze(2), epi_mask.unsqueeze(1))
                cdr3_alpha_mask_ = torch.logical_and(segment_mask.eq(3).unsqueeze(2), epi_mask.unsqueeze(1))
                cdr3_beta_mask_ = torch.logical_and(segment_mask.eq(6).unsqueeze(2), epi_mask.unsqueeze(1))
                
                pred_dist.extend(out[:, :, :, 0].detach().cpu().numpy().tolist())
                pred_contact.extend(out[:, :, :, 1].detach().cpu().numpy().tolist())
                cdr1_alpha_mask.extend(cdr1_alpha_mask_.detach().cpu().numpy().tolist())
                cdr3_alpha_mask.extend(cdr3_alpha_mask_.detach().cpu().numpy().tolist())
                cdr3_beta_mask.extend(cdr3_beta_mask_.detach().cpu().numpy().tolist())
                
                epi.extend(batch['epitope_seq'])
                mhc.extend(batch['mhc_allele'])
                cdr1_alpha.extend(batch['cdr1_alpha_seq'])
                cdr3_alpha.extend(batch['cdr3_alpha_seq'])
                cdr3_beta.extend(batch['cdr3_beta_seq'])
        
        res = []
        for i in range(len(epi)):
            epi_, mhc_ = epi[i], mhc[i]
            cdr1_alpha_, cdr3_alpha_, cdr3_beta_ = cdr1_alpha[i], cdr3_alpha[i], cdr3_beta[i]
            cdr1_alpha_mask_, cdr3_alpha_mask_, cdr3_beta_mask_ = np.array(cdr1_alpha_mask[i]), np.array(cdr3_alpha_mask[i]), np.array(cdr3_beta_mask[i])
            dist, contact = np.array(pred_dist[i]), np.array(pred_contact[i])
            dist_cdr1_alpha, contact_cdr1_alpha = dist[cdr1_alpha_mask_].reshape(len(cdr1_alpha_), len(epi_)), contact[cdr1_alpha_mask_].reshape(len(cdr1_alpha_), len(epi_))
            dist_cdr3_alpha, contact_cdr3_alpha = dist[cdr3_alpha_mask_].reshape(len(cdr3_alpha_), len(epi_)), contact[cdr3_alpha_mask_].reshape(len(cdr3_alpha_), len(epi_))
            dist_cdr3_beta, contact_cdr3_beta = dist[cdr3_beta_mask_].reshape(len(cdr3_beta_), len(epi_)), contact[cdr3_beta_mask_].reshape(len(cdr3_beta_), len(epi_))
            res.append({'epitope': epi_, 'mhc': mhc_, 'cdr1.alpha': cdr1_alpha_, 'cdr3.alpha': cdr3_alpha_, 'cdr3.beta': cdr3_beta_,
                        'dist': {'cdr1.alpha': dist_cdr1_alpha, 'cdr3.alpha': dist_cdr3_alpha, 'cdr3.beta': dist_cdr3_beta},
                        'contact': {'cdr1.alpha': contact_cdr1_alpha, 'cdr3.alpha': contact_cdr3_alpha, 'cdr3.beta': contact_cdr3_beta}})
        
        with open(os.path.join(self.log_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump(res, f)
        