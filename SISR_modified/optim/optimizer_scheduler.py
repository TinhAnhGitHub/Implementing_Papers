from torch.optim import Optimizer
from typing import Optional, List, Dict, Any
import math
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
)
import torch
from torch.optim import SGD, Adam, AdamW

class WarmupScheduler(_LRScheduler):
    """
    Wrapper for learning rate scheduler with multiple strategy

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        scheduler (_LRScheduler): Base scheduler to wrap.
        warmup_epochs (int): Number of warmup epochs.
        warmup_strategy (str): Warmup strategy ('linear', 'cos', 'constant'). Defaults to 'linear'.
        warmup_bias_lr (float, optional): Initial learning rate for bias parameters.
        warmup_momentum (float, optional): Initial momentum value.
        momentum (float, optional): Final momentum value.
        init_lr (float, optional): Initial learning rate. Defaults to 1e-3.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        warmup_epochs: int,
        warmup_strategy: str = 'linear',
        warmup_bias_lr: Optional[float] = None,
        warmup_momentum: Optional[float] = None,
        momentum: Optional[float] = None,
        init_lr: float = 1e-3,
        last_epoch: int = -1,
    ):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError(
                f"Expected warmup_strategy to be one of ['linear', 'cos', 'constant'] "
                f"but got {warmup_strategy}"
            )
    
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_strategy = warmup_strategy
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.momentum = momentum
        self.init_lr = init_lr

        self._warmup_func  = {
            'cos': self._warmup_cos,
            'linear': self._warmup_linear,
            'constant': self._warmup_const
        }[warmup_strategy]

        super().__init__(optimizer, last_epoch)
        self._format_param()
    
    def _format_param(self):
        for group in self.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self.init_lr, group['lr'])
        
    def _warmup_cos(self, start:float, end:float, percentage: float) -> float:
        cos_out = math.cos(math.pi * percentage) + 1
        return end + (start - end) / 2.0 * cos_out
    
    def _warmup_const(self, start:float, end:float, percentage: float) -> float:
        return start if percentage < 0.9999 else end

    def _warmup_linear(self, start:float, end:float, percentage: float) -> float:
        return (end-start) * percentage + start
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs

            lrs = []
            for group in self.optimizer.param_groups:
                if 'warmup_max_lr' not in group:
                    self._format_param()
                
                computed_lr = self._warmup_func(
                    group['warmup_initial_lr'],
                    group['warmup_max_lr'],
                    alpha
                )
                if self.warmup_bias_lr is not None and getattr(group.get('params', [None])[0], 'bias', False):
                    scale = alpha * (1 - self.warmup_bias_lr) + self.warmup_bias_lr
                    computed_lr *= scale
                lrs.append(computed_lr)
        else:
            return self.scheduler.get_lr()
    
    def step(self, epoch: Optional[int] = None) -> None:
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        
            if hasattr(self.optimizer, 'momentum') and self.warmup_momentum is not None:
                alpha = self.last_epoch / self.warmup_epochs
                momentum = alpha * (self.momentum - self.warmup_momentum) + self.warmup_momentum
                for group in self.optimizer.param_groups:
                    group['momentum'] = momentum
        else:
            self.scheduler.step(epoch)
            self._last_lr = self.scheduler.get_last_lr()
    

    def state_dict(self) -> dict:
        wrapper_state_dict = {
            key: value for key, value in self.__dict__.items() 
            if key not in ('optimizer', 'scheduler')
        }
        wrapped_state_dict = {
            key: value for key, value in self.scheduler.__dict__.items() 
            if key != 'optimizer'
        }
        return {
            'wrapped': wrapped_state_dict,
            'wrapper': wrapper_state_dict
        }

    def load_state_dict(self, state_dict: dict) -> None:
        
        self.__dict__.update(state_dict['wrapper'])
        self.scheduler.__dict__.update(state_dict['wrapped'])

    def __getattr__(self, name: str):
        return getattr(self.scheduler, name)



def create_optimizer_scheduler(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    warmup_config: Dict[str, Any],
    total_steps: int,
):
    """
    Create optimizer and scheduler with warmup support.

    Args:
        model (torch.nn.Module): Model parameters to optimize.
        optimizer_config (dict): Optimizer configuration.
        scheduler_config (dict): Scheduler configuration.
        warmup_config (dict): Warmup configuration.
        total_steps (int): Total number of training steps.

    Returns:
        Tuple[Optimizer, _LRScheduler]: The optimizer and scheduler.
    """
    optimizer_type = optimizer_config["type"].lower()
    lr = float(optimizer_config["lr"])
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if optimizer_type == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optimizer_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get("nesterov", False),
        )
    elif optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    scheduler_type = scheduler_config["type"].lower()
    if scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=int(scheduler_config["step_size"]),
            gamma=float(scheduler_config["gamma"]),
        )
    elif scheduler_type == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=float(scheduler_config["gamma"]))
    elif scheduler_type == "cosine":
        lr_end = float(scheduler_config["lr_end"])
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_end)
    elif scheduler_type == "plateau":
        lr_end = float(scheduler_config["lr_end"])
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=lr_end,
        )
    elif scheduler_type == "cyclic":
        lr_min = float(scheduler_config["lr_min"])
        lr_max = float(scheduler_config["lr_max"])
        cycle_steps = int(scheduler_config["cycle_steps"])
        scheduler = CyclicLR(
            optimizer,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=cycle_steps // 2,
            mode=scheduler_config["mode"],
        )
    elif scheduler_type == "one_cycle":
        lr_max = float(scheduler_config["lr_max"])
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr_max,
            total_steps=total_steps,
            pct_start=float(scheduler_config["pct_start"]),
            div_factor=float(scheduler_config["div_factor"]),
            final_div_factor=float(scheduler_config["final_div_factor"]),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    if float(warmup_config.get("epochs", 0)) > 0:
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_epochs=int(warmup_config["epochs"]),
            warmup_bias_lr=warmup_config.get("bias_lr"),
            warmup_momentum=warmup_config.get("momentum"),
            momentum=warmup_config.get("final_momentum"),
            warmup_strategy=warmup_config["strategy"],
        )

    return optimizer, scheduler