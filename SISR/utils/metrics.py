from typing import Dict, Any, Optional, Union, List,  Type, Literal
from torch import Tensor
from omegaconf import DictConfig
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import torch
from .train_utils import MetricTracker
    
class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, name: str, metric_cls: Type[Any]) -> None:
        cls._metrics[name] = metric_cls
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Any]]:
        return cls._metrics.get(name)

    @classmethod
    def available_metrics(cls) -> List[str]:
        return list(cls._metrics.keys())
    



class PSNRMetric:
    def __init__(self, data_range: float = 1.0, device: Optional[torch.device] = None):
        self.psnr = PSNR(data_range=data_range).to(device)
        self.tracker = MetricTracker(name="PSNR", mode="max", fmt="{:.2f}")
        self.device = device
    
    def compute(self, preds: Tensor, targets: Tensor) -> float:
        if self.device:
            preds = preds.to(self.device)
            targets = targets.to(self.device)
            
        value = self.psnr(preds, targets).item()
        self.tracker.update(value)
        return value


    def reset(self) -> None:
        self.psnr.reset()
        self.tracker.reset()

    def to(self, device: torch.device) -> 'PSNRMetric':
          self.device = device
          self.psnr = self.psnr.to(device)
          return self



class SSIMMetric:
    def __init__(self, data_range: float = 1.0, device: Optional[torch.device] = None):
        self.ssim = SSIM(data_range=data_range).to(device)
        self.tracker = MetricTracker(name="SSIM", mode="max", fmt="{:.4f}")
        self.device = device
    
    def compute(self, preds: Tensor, targets: Tensor) -> float:
        if self.device:
            preds = preds.to(self.device)
            targets = targets.to(self.device)
        value = self.ssim(preds, targets).item()
        self.tracker.update(value)
        return value


    def reset(self) -> None:
        self.ssim.reset()
        self.tracker.reset()
    
    def to(self, device: torch.device) -> 'SSIMMetric':
          self.device = device
          self.ssim = self.ssim.to(device)
          return self

class LossMetric:
    def __init__(self, device):
        self.tracker = MetricTracker(name="Loss", mode="min", fmt="{:.4f}")
        self._total = 0.0
        self._count = 0
    
    def update(self, loss: float, batch_size: int = 1) -> None:
        self._total += loss * batch_size
        self._count += batch_size
        
        if self._count > 0:
            avg_loss = self._total / self._count
            self.tracker.update(avg_loss)
    
    def reset(self) -> None:
        self._total = 0.0
        self._count = 0
        self.tracker.reset()


MetricRegistry.register(name="psnr", metric_cls=PSNRMetric)
MetricRegistry.register(name="ssim", metric_cls=SSIMMetric)
MetricRegistry.register(name="loss", metric_cls=LossMetric)

import torch

class MetricCollection:
    def __init__(
        self,
        config: DictConfig,
        metric_type: Literal['train', 'val'],
        device: Optional[torch.device] = None
    ):
        self.metric_type = metric_type
        self.metrics: Dict[str, Any] = {}
        self.device = device
        self._initialize_metrics(config)

    def _initialize_metrics(self, config: DictConfig) -> None:
        print("DEVICEEEEEEEEEEE", self.device)
        metrics_config = getattr(config.metrics, self.metric_type, {})
        for metric_name in metrics_config:
            metric_cls = MetricRegistry.get(metric_name)
            if metric_cls is None:
                raise ValueError(f"Unknown metric: {metric_name}")
            self.metrics[metric_name] = metric_cls(device=self.device)

    def update(self, metrics_values: Dict[str, Union[float, Tensor]]) -> None:
        for metric_name, value in metrics_values.items():
            metric = self.metrics.get(metric_name)
            
            if metric is None:
                raise ValueError(f"Metric {metric_name} is not registered.")
            
            
            if hasattr(metric, "compute"):
                metric.compute(value[0], value[1])  
        
            if hasattr(metric, "update"):
                metric.update(value)

    def get_summary_history(self):
        return {
            name: metric.tracker.history for name, metric in self.metrics.items()
        }
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: metric.tracker.get_summary()
            for name, metric in self.metrics.items()
        }

    def get_best_values(self) -> Dict[str, float]:
        return {
            name: metric.tracker.best_value
            for name, metric in self.metrics.items()
        }

    def get_improvements(self, window: int = 1) -> Dict[str, float]:
        return {
            name: metric.tracker.get_improvement(window)
            for name, metric in self.metrics.items()
        }
