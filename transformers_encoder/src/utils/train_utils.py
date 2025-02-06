import math
import os
import random
from omegaconf import OmegaConf
from datetime import timedelta
from typing import Union, Dict

import numpy as np
import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, nvmlShutdown, NVML_TEMPERATURE_GPU

import wandb


import torch.distributed as dist
import torch.multiprocessing as mp




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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def init_wandb(config: dict) :
    """
    Initialize a Weights & Biases (wandb) run with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing wandb settings.
    
    Returns:
        wandb.Run: Initialized wandb run object.
    
    Notes:
        - Supports both full dataset and fold-specific run naming
        - Uses anonymous mode for logging
        - Sets project, tags, and run name from config
    """
    project = config["wandb"]["project"]
    tags = config["tags"]
    
    
    run_id = f"{config['wandb']['run_name']}-all-data"
    
    return wandb.init(
        project=project,
        config=config,
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
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



def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Retrieve the current learning rate from an optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to get the learning rate from.
    
    Returns:
        float: Current learning rate multiplied by 1e6 for easier reading.
    """
    return optimizer.param_groups[0]['lr'] * 1e6




class AverageMeter:
    def __init__(self) -> None:
        self.reset()
        self.history = []
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(self.avg)





    
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    
    def register(self) -> None:
        """Create initial shadow copies of model parameters

        Raises:
            ValueError: _description_
        """
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()
        for name, buf in self.model.named_buffers(): # tensors that are not registered as model's parameters
            self.shadow[name] = buf.data.clone()
        

    def update(self) -> None:
        """Update shadow parameters uisng Exponential Average Moving

        Applies decay to blend current and previous parameter values
        new_weight = previous & decay + current * (1.0 - decay)

        Raises:
            ValueError: _description_
        """
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()
        for name, buf in self.model.named_buffers():
            self.shadow[name] = buf.data.clone()

    
    def apply_shadow(self) -> None:
        """Replace model's parameters with their shadow copies

        store the model's parameters in the backup dict, before replacement

        Raises:
            ValueError: _description_
        """
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name].to(param.device)
        for name, buf in self.model.named_buffers():
            self.backup[name] = buf.data
            buf.data = self.shadow[name].to(buf.device)

    def restore(self) -> None:
        """
        Restore original model parameters from backup.
        
        Clears the backup dictionary after restoration.
        """
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].to(param.device)
        for name, buf in self.model.named_buffers():
            buf.data = self.backup[name].to(buf.device)
        self.backup = {}




class AWP:
    """Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, criterion, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}
        self.criterion = criterion

    def attack_backward(self, batch, accelerator):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        outputs  = self.model(
            X=batch["input_ids"],
            valid_lens=batch["valid_lengths"]
        )
        labels = batch["labels"]
        adv_loss = self.criterion(outputs, labels)  
        adv_loss = adv_loss.mean()
        self.optimizer.zero_grad()
        accelerator.backward(adv_loss)
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}







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
