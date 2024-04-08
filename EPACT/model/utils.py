import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyStopping:

    def __init__(self, patience=10, checkpoint_dir='logs'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), os.path.join(
            self.checkpoint_dir, 'checkpoint.pt'))
        self.best_score = score


def unfreeze_params(modules: List):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        