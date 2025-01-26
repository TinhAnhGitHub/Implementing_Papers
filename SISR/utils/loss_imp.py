"""
Implementing Loss 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import Optional

class PatchKernel(nn.Module):
    """
    Implements the Image Patch Kernel (IPK) and Feature Patch Kernel(FPK)
    from section 3.1 of the paper: https://www.sciencedirect.com/science/article/pii/S0031320323002108#sec0005
    """

    def __init__(self, kernel_size: int, stride: Optional[int]=None, in_channels: int=1) -> None:
        """
        Args:
            kernel_size: Size of the patch (p in paper)
            stride: Stride for patch extraction (s in paper)
            in_channels: Number of input channels (default=1 for Y channel)
        """
        super().__init__()
        stride = stride or (kernel_size//2+1)
        
        kernel = torch.eye(
            kernel_size**2
        ).view(kernel_size**2, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(1, in_channels, 1, 1)  
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(kernel_size ** 2), requires_grad=False)
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Convert image/feature into patches using convolution.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Patches of shape [B, N, P*P] where N is number of patches
            and P is patch_size
        """
        batch_size = x.size(0)
        print(x.shape)
        patches = F.conv2d(
            x, self.weight, self.bias, 
            stride=self.stride, 
            padding=0
        )
        patches = patches.permute(
            0,2,3,1
        ).reshape(batch_size, -1, self.kernel_size**2) # [B, numpatches, patch_dimension]
    

        return patches
    
class TextureComplexity(nn.Module):
    """
    Implements the textural complexity lambda calculation
    """
    def __init__(self, eps: float=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Calculate textural complexity for each patch
        Args:
            patches: Tensor of shape [B, N, P*P] where N is number of patches

        Returns:
            Complexity scores of shape [B, N]
        """
        patch_mean = patches.mean(dim=-1, keepdim=True)
        L = patches.size(-1)
        complexity = torch.sqrt(
            ((patches-patch_mean)**2).sum(dim=-1) / (L-1 + self.eps)
        )
        return complexity
    

class AdaptiveAttention(nn.Module):
    """
    Implements the adaptive patch-wise attention
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.texture_complexity = TextureComplexity(eps)
        self.eps = eps
    
    def forward(self, patches:torch.Tensor) -> torch.Tensor:
        """
        Cal the attention weights for patches based on texture complexity
        Args:
            patches: Tensor of shape [B, N, P*P]

        Returns:
            Attention weights of shape [B, N]
        """

        complexity = self.texture_complexity(patches)
        attention = complexity / (complexity.sum(
            dim=-1, keepdim=True
        )+self.eps)
        return attention

class PatchLoss(nn.Module):
    """Complete implementation of the patch loss with multi-scale processing
    and adaptive attention as described in the paper.
    """
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config

        self.image_kernels = nn.ModuleList([
            PatchKernel(size, in_channels=1) 
            for size in config.patch_loss_config.image_patch_sizes
        ])

        self.feature_kernels = nn.ModuleList([
            PatchKernel(size, in_channels=512) 
            for size in config.patch_loss_config.feature_patch_sizes
        ])
        
        self.attention = AdaptiveAttention(config.patch_loss_config.eps)
    
    def compute_cosine_distance(
        self, x:torch.Tensor, y:torch.Tensor
    ):
        """
        Compute cosine distance between patches 
        Args:
            x: SR patches [B, N, P*P]
            y: HR patches [B, N, P*P]

        Returns:
            Cosine distance for each patch [B, N]
        """
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)

        cos_sim = (x_norm * y_norm).sum(dim=-1)

        distance = 1 - cos_sim
    
        return distance

    def compute_scale_loss(
        self,
        sr:torch.Tensor,
        hr:torch.Tensor,
        kernel: PatchKernel
    )-> torch.Tensor:
        """
        Compute loss for a single scale with attention weighting
        """

        sr_patches = kernel(sr)
        hr_patches = kernel(hr)

        distance = self.compute_cosine_distance(
            sr_patches,hr_patches
        )
        attention = self.attention(hr_patches)

        weight_loss = (distance*attention).sum()
        return weight_loss

    def _move_kernels_to_device(self, device):
        for kernel in self.image_kernels:
             kernel.weight.data = kernel.weight.to(device)
             kernel.bias.data = kernel.bias.to(device)
        for kernel in self.feature_kernels:
             kernel.weight.data = kernel.weight.to(device)
             kernel.bias.data = kernel.bias.to(device)

    def forward(
        self,
        sr:torch.Tensor,
        hr:torch.Tensor,
        lr_features: Optional[torch.Tensor]=None,
        hr_features: Optional[torch.Tensor]=None
    ):
        """
        Compute total patch loss with multi-scale processing and attention

        Args:
            sr: Super-resolved image [B, C, H, W]
            hr: High-resolution ground truth [B, C, H, W]
            sr_features: Optional features from SR network
            hr_features: Optional features from HR network

        Returns:
            Total loss value
        """        
        self._move_kernels_to_device(sr.device)
        sr_y = 16. + (65.481 * sr[:, 0] + 128.553 * sr[:, 1] + 24.966 * sr[:, 2])/255.
        hr_y = 16. + (65.481 * hr[:, 0] + 128.553 * hr[:, 1] + 24.966 * hr[:, 2])/255.
        sr_y = sr_y.unsqueeze(1)
        hr_y = hr_y.unsqueeze(1)

        image_loss = sum(
            self.compute_scale_loss(sr_y, hr_y, kernel) for kernel in self.image_kernels
        )
        feature_loss = 0.0

        if lr_features is not None and hr_features is not None:
            feature_loss = sum(
                self.compute_scale_loss(lr_features, hr_features, kernel)
                for kernel in self.feature_kernels
            )

        total_loss = image_loss + self.config.patch_loss_config.kappa * feature_loss
        return total_loss
