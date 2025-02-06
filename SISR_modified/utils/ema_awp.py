# utils/ema_awp.py
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, Any] = {}
        self.backup: dict[str, Any] = {}

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()
        for name, buf in self.model.named_buffers():
            self.shadow[name] = buf.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()
        for name, buf in self.model.named_buffers():
            self.shadow[name] = buf.data.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name].to(param.device)
        for name, buf in self.model.named_buffers():
            self.backup[name] = buf.data
            buf.data = self.shadow[name].to(buf.device)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].to(param.device)
        for name, buf in self.model.named_buffers():
            buf.data = self.backup[name].to(buf.device)
        self.backup = {}


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, *, adv_param: str = "weight", adv_lr: float = 0.001, adv_eps: float = 0.001
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup: dict[str, torch.Tensor] = {}

    def perturb(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param].get("exp_avg", None)
                if grad is None:
                    continue
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))
                    param.data.clamp_(param_min, param_max)

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
