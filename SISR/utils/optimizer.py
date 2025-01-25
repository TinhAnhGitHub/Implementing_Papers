# optimizers.py

from typing import Dict, Any, Optional, Union, Type, List
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass
from torchcontrib.optim import SWA


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: Optional[Dict[str, Any]] = None

@dataclass
class SchedulerConfig:
    name: str
    warmup_epochs: int = 0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    min_lr: float = 1e-6
    cycle_momentum: bool = True

class OptimizerFactory:
    

    _optimizers: Dict[str, Type[Optimizer]] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD
    }

    @classmethod
    def register_optimizer(cls, name: str, optimizer_cls: Type[Optimizer]) -> None:
        
        cls._optimizers[name.lower()] = optimizer_cls

    @classmethod
    def create(
        cls,
        params,
        config: Union[OptimizerConfig, Dict[str, Any]],
        base_lr: int
    ) -> Optimizer:

        if isinstance(config, dict):
            config = OptimizerConfig(**config)

        optimizer_cls = cls._optimizers.get(config.name.lower())
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {config.name}")


        kwargs = cls._get_optimizer_kwargs(optimizer_cls, config, base_lr)

        optimizer = optimizer_cls(params, **kwargs)
        if config.grad_clip and config.grad_clip.get("enable", False):
            for group in optimizer.param_groups:
                group.setdefault(
                    "max_norm", config.grad_clip.get("max_norm", 1.0)
                )
                group.setdefault(
                    "norm_type", config.grad_clip.get("norm_type", 2.0)
                )

        return optimizer

    @staticmethod
    def _get_optimizer_kwargs(optimizer_cls: Type[Optimizer], config: OptimizerConfig, base_lr:int) -> Dict[str, Any]:
        kwargs = {"lr": base_lr}

        if optimizer_cls in [optim.Adam, optim.AdamW]:
            kwargs.update({
                "betas": config.betas,
                "eps": config.eps,
                "weight_decay": config.weight_decay
            })
        elif optimizer_cls == optim.SGD:
            kwargs.update({
                "momentum": config.momentum,
                "weight_decay": config.weight_decay
            })

        return kwargs

class LRSchedulerFactory:

    _schedulers: Dict[str, Type[_LRScheduler]] = {
        "cosineannealinglr": optim.lr_scheduler.CosineAnnealingLR,
        "cosineannealingWarmrestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "onecyclelr": optim.lr_scheduler.OneCycleLR,
    }

    @classmethod
    def register_scheduler(cls, name: str, scheduler_cls: Type[_LRScheduler]) -> None:
        cls._schedulers[name.lower()] = scheduler_cls

    @classmethod
    def create(
        cls,
        optimizer: Optimizer,
        config: Union[SchedulerConfig, Dict[str, Any]],
        num_epochs: int,
        steps_per_epoch: Optional[int] = None
    ) -> _LRScheduler:
        if isinstance(config, dict):
            config = SchedulerConfig(**config)

        scheduler_cls = cls._schedulers.get(config.name.lower())
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler: {config.name}")

        kwargs = cls._get_scheduler_kwargs(
            scheduler_cls,
            config,
            num_epochs,
            steps_per_epoch
        )

        return scheduler_cls(optimizer, **kwargs)

    @staticmethod
    def _get_scheduler_kwargs(
        scheduler_cls: Type[_LRScheduler],
        config: SchedulerConfig,
        num_epochs: int,
        steps_per_epoch: Optional[int]
    ) -> Dict[str, Any]:
        
        kwargs = {}

        if scheduler_cls == optim.lr_scheduler.CosineAnnealingLR:
            kwargs.update({
                "T_max": num_epochs,
                "eta_min": config.min_lr
            })
        elif scheduler_cls == optim.lr_scheduler.CosineAnnealingWarmRestarts:
            kwargs.update({
                "T_0": config.warmup_epochs,
                "T_mult": 2,
                "eta_min": config.min_lr
            })
        elif scheduler_cls == optim.lr_scheduler.OneCycleLR:
            assert steps_per_epoch is not None, "steps_per_epoch required for OneCycleLR"
            kwargs.update({
                "max_lr": config.warmup_bias_lr,
                "epochs": num_epochs,
                "steps_per_epoch": steps_per_epoch,
                "pct_start": config.warmup_epochs / num_epochs,
                "cycle_momentum": config.cycle_momentum
            })

        return kwargs

class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: _LRScheduler,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_bias_lr * alpha for _ in self.base_lrs]

        return self.base_scheduler.get_lr()


def create_optimizer_and_scheduler(model: torch.nn.Module, config: Dict[str, Any], base_lr:int, steps_per_epoch:int):
    """Create optimizer and scheduler from config"""

    optimizer = OptimizerFactory.create(
        model.parameters(),
        config.optimizer,
        base_lr
    )


    scheduler = LRSchedulerFactory.create(
        optimizer,
        config.training.lr_scheduler,
        num_epochs=config.training.num_epochs,
        steps_per_epoch=steps_per_epoch
    )

    if config.training.lr_scheduler.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=config.training.lr_scheduler.warmup_epochs,
            base_scheduler=scheduler,
            warmup_momentum=config.training.lr_scheduler.warmup_momentum,
            warmup_bias_lr=config.training.lr_scheduler.warmup_bias_lr
        )

    return optimizer, scheduler
