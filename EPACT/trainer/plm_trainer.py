import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

from .base import BaseTrainer
from ..model import PeptideLM, TCRCDR3LM, TCRCDR123LM
from ..model.utils import EarlyStopping


class EpitopeLMTrainer(BaseTrainer):
    def __init__(self, config):
        super(EpitopeLMTrainer, self).__init__(config)

        self.model = PeptideLM(**config.model)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
    def proceed_one_epoch(self, iterator, training=True):

        avg_mlm_loss = 0
        avg_atchely_loss = 0
        avg_acc = 0
        data_size = 0
        if training:
            self.model.train()
        else:
            self.model.eval()

        for batch in tqdm(iterator):
            
            toks = batch['token'].to(self.device)
            labels = batch['label'].to(self.device)
            res = self.model(toks)
            logits = res['logits']

            mask = (toks == self.tokenizer.mask_idx)
            mask_logits = logits[mask]
            mask_labels = labels[mask]

            mlm_loss = self.loss_fn(mask_logits, mask_labels)
            mask_preds = torch.argmax(mask_logits, dim=1)
            acc = accuracy_score(mask_labels.cpu().numpy(),
                                 mask_preds.cpu().numpy())
            
            if 'atchely_factor' in batch.keys():
                atchely_factor = batch['atchely_factor'].to(self.device)
                pred_atchely_factor = res['atchely_factor']
                atchely_factor_loss = F.mse_loss(pred_atchely_factor[mask], atchely_factor[mask])
                loss = mlm_loss + 0.2 * atchely_factor_loss
                avg_atchely_loss += atchely_factor_loss.cpu().item() * len(mask_labels)
            else:
                loss = mlm_loss

            avg_mlm_loss += mlm_loss.cpu().item() * len(mask_labels)
            avg_acc += acc * len(mask_labels)
            data_size += len(mask_labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        avg_mlm_loss = avg_mlm_loss / data_size
        avg_perplexity = np.exp(avg_mlm_loss)
        avg_acc = avg_acc / data_size
        avg_atchely_loss = avg_atchely_loss / data_size

        return avg_mlm_loss, avg_perplexity, avg_acc, avg_atchely_loss

    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)

    def evaluate_one_epoch(self, eval_loader):
        return self.proceed_one_epoch(eval_loader, training=False)

    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_mlm_loss, train_perplexity, train_acc, train_mse_loss = self.train_one_epoch(train_loader)
            val_mlm_loss, val_perplexity, val_acc, val_mse_loss = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train MLM Loss: {train_mlm_loss:.3f} | Perplexity: {train_perplexity:.2f} | ACC: {train_acc*100:.2f}% | MSE Loss: {train_mse_loss:.3f}')
            self.logger.info(f'Valid MLM Loss: {val_mlm_loss:.3f} | Perplexity: {val_perplexity:.2f} | ACC: {val_acc*100:.2f}% | MSE Loss: {val_mse_loss:.3f}')

            self.writer.add_scalar('Train/Loss', train_mlm_loss, epoch+1)
            self.writer.add_scalar('Train/Perplexity', train_perplexity, epoch+1)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
            self.writer.add_scalar('Valid/Loss', val_mlm_loss, epoch+1)
            self.writer.add_scalar('Valid/Perplexity', val_perplexity, epoch+1)
            self.writer.add_scalar('Valid/Accuracy', val_acc, epoch+1)

            self.early_stopping(val_perplexity, self.model, goal="minimize")
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
            
        self.writer.close()

    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        test_mlm_loss, test_perplexity, test_acc, test_mse_loss = self.evaluate_one_epoch(test_loader)
        self.logger.info(f'Test MLM Loss: {test_mlm_loss:.3f} | Perplexity: {test_perplexity:.2f} | ACC: {test_acc*100:.2f}% | MSE Loss: {test_mse_loss:.3f}')

    def predict(self, data_loader, model_location):
        pass


class PairedTCRLMTrainer(BaseTrainer):
    def __init__(self, config):
        super(PairedTCRLMTrainer, self).__init__(config)

        if config.data.use_cdr123:
            self.model = TCRCDR123LM(**config.model)
        else:
            self.model = TCRCDR3LM(**config.model)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
    def proceed_one_epoch(self, iterator, training=True):
        
        avg_mlm_loss = 0
        avg_atchely_loss = 0
        avg_acc = 0
        data_size = 0
        
        if training:
            self.model.train()
        else:
            self.model.eval()

        for batch in tqdm(iterator):
            
            toks = batch['token'].to(self.device)
            chain_mask = batch['chain_token_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            res = self.model(toks, chain_mask)
            logits = res['logits']

            mask = (toks == self.model.tokenizer.mask_idx)
            mask_logits = logits[mask]
            mask_labels = labels[mask]

            mlm_loss = self.loss_fn(mask_logits, mask_labels)
            mask_preds = torch.argmax(mask_logits, dim=1)
            acc = accuracy_score(mask_labels.cpu().numpy(),
                                 mask_preds.cpu().numpy())
            
            if 'atchely_factor' in batch.keys():
                atchely_factor = batch['atchely_factor'].to(self.device)
                pred_atchely_factor = res['atchely_factor']
                atchely_factor_loss = F.mse_loss(pred_atchely_factor[mask], atchely_factor[mask])
                loss = mlm_loss + 0.2 * atchely_factor_loss
                avg_atchely_loss += atchely_factor_loss.cpu().item() * len(mask_labels)
            else:
                loss = mlm_loss

            avg_mlm_loss += mlm_loss.cpu().item() * len(mask_labels)
            avg_acc += acc * len(mask_labels)
            data_size += len(mask_labels)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        avg_mlm_loss = avg_mlm_loss / data_size
        avg_perplexity = np.exp(avg_mlm_loss)
        avg_acc = avg_acc / data_size
        avg_atchely_loss = avg_atchely_loss / data_size

        return avg_mlm_loss, avg_perplexity, avg_acc, avg_atchely_loss

    def train_one_epoch(self, train_loader):
        return self.proceed_one_epoch(train_loader, training=True)

    def evaluate_one_epoch(self, eval_loader):
        return self.proceed_one_epoch(eval_loader, training=False)

    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_mlm_loss, train_perplexity, train_acc, train_mse_loss = self.train_one_epoch(train_loader)
            val_mlm_loss, val_perplexity, val_acc, val_mse_loss = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train MLM Loss: {train_mlm_loss:.3f} | Perplexity: {train_perplexity:.2f} | ACC: {train_acc*100:.2f}% | MSE Loss: {train_mse_loss:.3f}')
            self.logger.info(f'Valid MLM Loss: {val_mlm_loss:.3f} | Perplexity: {val_perplexity:.2f} | ACC: {val_acc*100:.2f}% | MSE Loss: {val_mse_loss:.3f}')

            self.writer.add_scalar('Train/Loss', train_mlm_loss, epoch+1)
            self.writer.add_scalar('Train/Perplexity', train_perplexity, epoch+1)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
            self.writer.add_scalar('Valid/Loss', val_mlm_loss, epoch+1)
            self.writer.add_scalar('Valid/Perplexity', val_perplexity, epoch+1)
            self.writer.add_scalar('Valid/Accuracy', val_acc, epoch+1)

            self.early_stopping(val_perplexity, self.model, goal="minimize")
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
            
        self.writer.close()

    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location))
        test_mlm_loss, test_perplexity, test_acc, test_mse_loss = self.evaluate_one_epoch(test_loader)
        self.logger.info(f'Test MLM Loss: {test_mlm_loss:.3f} | Perplexity: {test_perplexity:.2f} | ACC: {test_acc*100:.2f}% | MSE Loss: {test_mse_loss:.3f}')

    def predict(self, data_loader, model_location):
        pass
