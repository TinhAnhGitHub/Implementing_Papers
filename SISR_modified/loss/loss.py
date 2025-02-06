"""
The implementation of Gradient Loss: https://arxiv.org/abs/1911.09428
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MixGradientLoss(nn.Module):
    def __init__(self, lambda_g: float):
        super().__init__()
        gx_kernel = torch.tensor([[-1., -2., -1.],
                                  [0.,  0.,  0.],
                                  [1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0)

        gy_kernel = torch.tensor([[-1.,  0.,  1.],
                                  [-2.,  0.,  2.],
                                  [-1.,  0.,  1.]]).unsqueeze(0).unsqueeze(0)

        self.weight1 = nn.Parameter(gx_kernel, requires_grad=False)  # Sobel X
        self.weight2 = nn.Parameter(gy_kernel, requires_grad=False)  # Sobel Y
        self.lambda_g = lambda_g

    def compute_gradient(self, img: torch.Tensor, weight: torch.Tensor):
        channels = [F.conv2d(img[:, i].unsqueeze(1), weight, padding=1) for i in range(3)]
        return torch.cat(channels, dim=1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(prediction, target)

        grad_x_pred = self.compute_gradient(prediction, self.weight1)
        grad_y_pred = self.compute_gradient(prediction, self.weight2)
        grad_x_target = self.compute_gradient(target, self.weight1)
        grad_y_target = self.compute_gradient(target, self.weight2)

        grad_pred = torch.sqrt(grad_x_pred ** 2 + grad_y_pred ** 2)
        grad_target = torch.sqrt(grad_x_target ** 2 + grad_y_target ** 2)

        gradient_loss = F.mse_loss(grad_pred, grad_target) / 10000.0

        mix_gradient_loss = mse_loss + self.lambda_g * gradient_loss

        return mix_gradient_loss



def loss_factory(loss_type: str, lambda_g: Optional[float]=None) -> nn.Module:
    if loss_type == "L1":
        return nn.L1Loss()
    elif loss_type == "L2":
        return nn.MSELoss()
    elif loss_type == "MixGradient":
        return MixGradientLoss(lambda_g)
    else:
        raise ValueError("Invalid loss type. Choose from 'L1', 'L2', or 'MixGradient'.")
