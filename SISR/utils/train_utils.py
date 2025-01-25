import os
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import torch 
import socket
from enum import Enum
from typing import Dict, Any, Optional, Union
import numpy as np
from contextlib import contextmanager
import random
import math
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, nvmlShutdown, NVML_TEMPERATURE_GPU
from omegaconf import DictConfig
import wandb




def init_wandb(config: DictConfig) :
    
    project = config.project
    tags = config.tags

    dir = config.dir
    run_id = f"{config.name}-all-data"
    
    return wandb.init(
        project=project,
        tags=tags,
        name=run_id,
        dir=dir,
        anonymous="allow",
        job_type="train",
    )


def log_gpu_metrics(gpu_index: int = 0) -> Dict[str, float]:
    """
    Collect detailed GPU metrics including memory usage, utilization, and temperature.

    Args:
        gpu_index (int): Index of the GPU to monitor (default is 0).

    Returns:
        Dict[str, float]: Dictionary containing GPU metrics.
    """
    metrics = {}
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        metrics['gpu_used_memory_mb'] = mem_info.used // 1024**2
        metrics['gpu_total_memory_mb'] = mem_info.total // 1024**2
        metrics['gpu_utilization'] = nvmlDeviceGetUtilizationRates(handle).gpu
        metrics['gpu_temperature'] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        nvmlShutdown()
    except Exception as e:
        print(f"Error while retrieving GPU metrics: {e}")
    return metrics



def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port: int = 5000, max_port: int = 5010) -> int:
    """Find an available port within specified range."""
    for port in range(start_port, max_port):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free ports in range {start_port}-{max_port}")


def as_minutes(seconds: Union[int, float]) -> str:
    if seconds < 0:
        raise ValueError("Time in seconds cannot be negative")

    days = math.floor(seconds/ 86400)
    seconds = seconds % 86400

    hours = math.floor(seconds / 3600)
    seconds = seconds % 3600

    minutes = math.floor(seconds / 60)
    remained_seconds = seconds % 60

    formatted_seconds = f"{remained_seconds:.2f}"

    time_components = []
    if days > 0:
        time_components.append(f"{days:.2f}")
    
    if hours > 0:
        time_components.append(f"{hours:.2f}")
    if minutes > 0:
        time_components.append(f"{minutes:.2f}")
    
    time_components.append(f"{formatted_seconds}s")

    return ':'.join(time_components)



@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


def setup(rank: int, world_size: int, timeout_seconds: float = 3600):
    try:
        timeout = timedelta(seconds=timeout_seconds)
    except TypeError as e:
        raise ValueError("timeout_seconds must be a valid float or integer.") from e

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '5553'

    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )

        torch.cuda.set_device(rank)
        print(f"Process {rank} initialized on GPU {rank} with timeout {timeout}.")
    except Exception as e:
        print(f"Failed to initialize process group for rank {rank}: {e}")
        raise


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cleanup_processes():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    
    for p in mp.active_children():
        p.terminate()
        p.join()
    
    print("Cleanup completed")




class MetricMode(Enum):
    MIN = "min"
    MAX = "max"


class MetricTracker():
    """
    Class for metrics that tracks values over time and provides statistics.
    """
    def __init__(
        self,
        name: str,
        mode: Union[str, MetricMode] = "max",
        higher_is_better: Optional[bool] = None,
        fmt: str = "{:.4f}",
        window_size: int = math.inf
    ):
        """
        Initialize the metric tracker.
        
        Args:
            name (str): Name of the metric
            mode (Union[str, MetricMode]): Optimization mode ('min' or 'max')
            higher_is_better (bool, optional): Whether higher values are better
            fmt (str): Format string for metric value display
            window_size (int): Maximum number of historical values to maintain
        """
        self.name = name
        self.mode = MetricMode(mode) if isinstance(mode, str) else mode
        self.higher_is_better = higher_is_better or (self.mode == MetricMode.MAX)
        self.fmt = fmt
        self.window_size = window_size
        
        self.history = []
        self.best_value = float("-inf") if self.higher_is_better else float("inf")
        self.worst_value = float("inf") if self.higher_is_better else float("-inf")
        self._is_active = True
        
        self.reset()
    
    def update(self, value: float) -> None:
        """
        Update metric with a new value.
        
        Args:
            value (float): New metric value to record
        """
        if not self._is_active:
            return
            
        self.history.append(value)
        
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
            
        if self.higher_is_better:
            self.best_value = max(self.best_value, value)
            self.worst_value = min(self.worst_value, value)
        else:
            self.best_value = min(self.best_value, value)
            self.worst_value = max(self.worst_value, value)
    
    def reset(self) -> None:
        self.history = []
        self.best_value = float("-inf") if self.higher_is_better else float("inf")
        self.worst_value = float("inf") if self.higher_is_better else float("-inf")
    
    
    
    def should_early_stop(self, patience: int = 5, min_delta: float = 0.0) -> bool:

        if len(self.history) < patience:
            return False
            
        best_in_window = max(self.history[-patience:]) if self.higher_is_better else min(self.history[-patience:])
        current = self.history[-1]
        
        if self.higher_is_better:
            return current < best_in_window - min_delta
        return current > best_in_window + min_delta
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
            
        return {
            "name": self.name,
            "current": self.history[-1] if self.history else None,
            "best": self.best_value,
            "worst": self.worst_value,
            "mean": np.mean(self.history).item(),
            "std": np.std(self.history).item(),
            "median": np.median(self.history).item(),
            "history_length": len(self.history),
            "formatted_current": self.fmt.format(self.history[-1]) if self.history else "N/A"
        }