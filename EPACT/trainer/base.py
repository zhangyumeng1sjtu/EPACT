from abc import abstractmethod
import logging
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from ..utils.tokenizer import IUPACTokenizer


class BaseTrainer(object):
    def __init__(self, config, log_dir=None):
        self.model = None
        self.tokenizer = IUPACTokenizer()
        self.device = torch.device(f"cuda:{config.training.gpu_device}" if torch.cuda.is_available() else "cpu")
        self.max_epochs = config.training.max_epochs
        
        if log_dir is None:
            self.log_dir = config.training.log_dir
        else:
            self.log_dir = log_dir

    def configure_optimizer(self, param_groups, training_config):
        optimizer = AdamW(param_groups, lr=training_config.lr, weight_decay=training_config.weight_decay)
        if training_config.lr_scheduler is None:
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=training_config.warm_epochs)
        else:
            if training_config.lr_scheduler == 'step':
                after_scheduler = StepLR(
                    optimizer, step_size=training_config.lr_decay_steps, gamma=training_config.lr_decay_rate)
            elif training_config.lr_scheduler == 'cosine':
                after_scheduler = CosineAnnealingLR(
                    optimizer, T_max=training_config.lr_decay_steps, eta_min=training_config.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=training_config.warm_epochs, after_scheduler=after_scheduler)
        return optimizer, scheduler

    def configure_logger(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = logging.getLogger(log_dir)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T")
        fh = logging.FileHandler(filename=os.path.join(
            log_dir, "training.log"), encoding='utf-8', mode='w+')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.writer = SummaryWriter(log_dir)

    @abstractmethod
    def proceed_one_epoch(self, iterator, training):
        raise NotImplementedError
    
    @abstractmethod
    def train_one_epoch(self, train_loader):
        raise NotImplementedError

    @abstractmethod
    def evaluate_one_epoch(self, eval_loader):
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_loader, val_loader):
        raise NotImplementedError

    @abstractmethod
    def test(self, test_loader, model_location):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_loader, model_location):
        raise NotImplementedError
    