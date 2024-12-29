"""
Loss functions for PyTorch.
"""
import torch
import torch.nn as nn

class UnifiedMaskRecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, cls_out, mask_out, mask_seq, x):
        """
        Compute the unified masked reconstruction loss
        
        Args:
            cls_out: classifier output for reconstruction [B, L, V]
            mask_out: masked sequence reconstruction [B, L, V]
            mask_seq: binary mask indicating masked positions [B, L]
            x: original input sequence [B, L, V]
        """
        # Compute loss only on masked positions
        mask_seq = mask_seq.unsqueeze(-1)  # [B, L, 1]
        
        # Reconstruction loss for masked positions
        mask_loss = self.criterion(mask_out * mask_seq, x * mask_seq)
        
        # Reconstruction loss for classifier output
        cls_loss = self.criterion(cls_out * mask_seq, x * mask_seq)
        
        # Combine losses
        total_loss = mask_loss + cls_loss
        
        return total_loss