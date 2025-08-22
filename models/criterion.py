import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" by Lin et al.
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float or tensor): Weighting factor for classes. If float, same weight for all.
                                If tensor, should have length equal to num_classes.
        gamma (float): Focusing parameter. Higher gamma puts more focus on hard examples.
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
        label_smoothing (float): Label smoothing factor (0.0 to 1.0)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs (tensor): Predicted logits of shape (N, C) where N is batch size, C is num_classes
            targets (tensor): Ground truth labels of shape (N,) with values in [0, C-1]
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + \
                             self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets_one_hot, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss that automatically calculates alpha weights
    based on class frequencies in the training data.
    
    Args:
        beta (float): Hyperparameter controlling the re-weighting strength (0.0 to 1.0)
        gamma (float): Focusing parameter for focal loss
        reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
        class_counts (tensor): Number of samples per class, used to calculate weights
    """
    
    def __init__(self, beta=0.999, gamma=2.0, reduction='mean', class_counts=None):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        
        if class_counts is not None:
            effective_num = 1.0 - torch.pow(self.beta, class_counts.float())
            weights = (1.0 - self.beta) / effective_num
            self.register_buffer('weights', weights / weights.sum() * len(weights))
        else:
            self.weights = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs (tensor): Predicted logits of shape (N, C)
            targets (tensor): Ground truth labels of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class balancing weights if available
        if self.weights is not None:
            alpha_t = self.weights[targets]
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that dynamically adjusts gamma based on training progress.
    Useful for gradually shifting focus from easy to hard examples during training.
    
    Args:
        alpha (float): Class weighting factor
        gamma_init (float): Initial gamma value
        gamma_final (float): Final gamma value
        reduction (str): Specifies the reduction to apply
    """
    
    def __init__(self, alpha=1.0, gamma_init=0.5, gamma_final=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.reduction = reduction
        self.current_gamma = gamma_init
        
    def update_gamma(self, epoch, max_epochs):
        """Update gamma based on training progress"""
        progress = epoch / max_epochs
        self.current_gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * progress
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.current_gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss