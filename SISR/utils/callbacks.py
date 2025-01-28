from typing import Any, Dict, Type, List, Optional, Set
from dataclasses import dataclass, field
from enum import IntEnum
from omegaconf import DictConfig
import inspect
from abc import ABC
import os
import torch
import numpy as np
from pathlib import Path
from torchcontrib.optim import SWA
import wandb
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from utils import as_minutes


class Priority(IntEnum):
    HIGHEST = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    LOWEST = 4

@dataclass
class CallbackConfig:
    name: str
    enable_bool: bool = True  
    priority: Priority = Priority.NORMAL
    params: Dict[str, Any] = field(default_factory=dict)



class Callback(ABC):
    def __init__(self, **kwargs):
        self.priority = kwargs.get("priority", Priority.NORMAL)

        valid_params = self._get_valid_parameters()
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @classmethod
    def _get_valid_parameters(cls) -> Set[str]:
        init_signature = inspect.signature(cls.__init__)
        return {
            param.name
            for param in init_signature.parameters.values()
            if param.name != "self" and param.name != "kwargs"
        }


    # Training lifecycle hooks
    def on_pretrain_routine_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger before pretraining routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after pretraining routine completes."""
        pass

    def on_train_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when training begins."""
        pass

    def on_train_epoch_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger at start of training epoch."""
        pass

    def on_train_batch_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger before processing training batch."""
        pass

    def after_forward_backward(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger during optimizer parameter update."""
        pass

    def before_forward_backward(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger before optimizer gradients reset."""
        pass

    def on_train_batch_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after processing training batch."""
        pass

    def on_train_epoch_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger at end of training epoch."""
        pass

    def on_fit_epoch_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after completing train+validation epoch."""
        pass

    def on_model_save(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when saving model checkpoint."""
        pass

    def on_train_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when training completes."""
        pass

    def on_params_update(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after parameter updates."""
        pass



    # Validation lifecycle hooks
    def on_val_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when validation begins."""
        pass

    def on_val_batch_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger before processing validation batch."""
        pass

    def on_val_batch_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after processing validation batch."""
        pass

    def on_val_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when validation completes."""
        pass

    # Prediction lifecycle hooks
    def on_predict_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when prediction begins."""
        pass

    def on_predict_batch_start(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger before processing prediction batch."""
        pass

    def on_predict_postprocess_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after prediction postprocessing."""
        pass

    def on_predict_batch_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger after processing prediction batch."""
        pass

    def on_predict_end(self, trainer: Any, params: Optional[Dict[str, Any]] = None) -> None:
        """Trigger when prediction completes."""
        pass

class CallbackManager:
    def __init__(self):
        self.callbacks: Dict[str, List[Callback]] = {}

    def add_callback(self, callback: Callback):
        
        event_names = self._get_callback_event_names()

        for event_name in event_names:
            callback_cls = type(callback)
            base_method = getattr(Callback, event_name)
            callback_method = getattr(callback_cls, event_name, None)

            
            if callback_method is not base_method:
                if event_name not in self.callbacks:
                    self.callbacks[event_name] = []
                self.callbacks[event_name].append(callback)

        for event_name in self.callbacks:
            self.callbacks[event_name].sort(key=lambda x: x.priority.value)
    
    def _get_callback_event_names(self) -> Set[str]:
        event_names = set()
        for name, method in inspect.getmembers(Callback, inspect.isfunction):
            if name.startswith("on_") or name.startswith("after_") or name.startswith("before_"):
                event_names.add(name)
        return event_names
    
    def trigger_event(self, event: str, *args, **kwargs):
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                getattr(callback, event)(*args, **kwargs)

   
class CallbackFactory:

    _registry: Dict[str, Type[Callback]] = {}
    _default_priorities: Dict[str, Priority] = {
    }

    @classmethod
    def register(cls, name: str, priority: Priority = None) -> None:
        def wrapper(callback_cls: Type[Callback]):
            cls._registry[name.lower()] = callback_cls
            if priority:
                cls._default_priorities[name.lower()] = priority
            return callback_cls
        return wrapper
    
        
    
    @classmethod
    def create_callback(
        cls, name:str, config: Dict[str, Any]
    ) -> Optional[Callback]:
        callback_cls = cls._registry.get(name.lower())
        if not callback_cls:
            raise ValueError(f"Unknown callback: {name}")

        params = {
            'priority': cls._default_priorities.get(name.lower()),
        }


        for key, value in config.items():
            if key not in ['enabled']:
                params[key] = value

        return callback_cls(**params)
    
    @classmethod
    def create_callback_manager(cls, config: DictConfig) -> CallbackManager:
        manager = CallbackManager()
        if not hasattr(config, 'callbacks'):
            return manager
        for name, callback_config in config.callbacks.items():
            if not callback_config.get('enabled'):
                continue
            callback = cls.create_callback(name, callback_config)
            if callback:
                manager.add_callback(callback=callback)

        return manager

    


@CallbackFactory.register(name="wandb", priority=Priority.LOWEST)
class WandbLoggerCallback(Callback):
    def __init__(self, log_train_metrics: bool = True, log_val_metrics: bool = True, log_grad_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.log_train_metrics = log_train_metrics
        self.log_val_metrics = log_val_metrics
        self.log_grad_norm = log_grad_norm


    def on_train_batch_end(self, trainer, params: Optional[Dict[str, Any]] = None):
        
        if trainer.rank == 0:
            global_step = len(trainer.train_loader) * trainer.state.current_epoch + trainer.state.current_batch_step
            log_dict = {}

            if self.log_train_metrics:
                train_metrics_summary = trainer.train_metrics.get_summary()
                log_dict.update(
                    {f"train/{k}": v['current'] for k, v in train_metrics_summary.items()}
                )
            
            
            if self.log_grad_norm and hasattr(trainer, 'gradient_norm_history') and trainer.gradient_norm_history:
                grad_stats = trainer.gradient_norm_history[-1]
                if 'global_gradient_norm'  in grad_stats:
                    log_dict.update({
                        'global_gradient_norm': grad_stats['global_gradient_norm']
                    })
                if log_dict:
                    wandb.log(log_dict, step=global_step)
            
        trainer.accelerator.wait_for_everyone()
            

    def on_val_batch_end(self, trainer, params: Optional[Dict[str, Any]] = None):
        if trainer.rank == 0:
            
            global_step = len(trainer.train_loader) * trainer.state.current_epoch + trainer.state.current_batch_step
    
            log_dict = {}
    
            if self.log_val_metrics:
                val_metrics_summary = trainer.val_metrics.get_summary()
                log_dict.update({f"val/{k}": v['current'] for k, v in val_metrics_summary.items()})
    
            if log_dict: 
                wandb.log(log_dict, step=global_step)

        trainer.accelerator.wait_for_everyone()
    

            


@CallbackFactory.register(name="modelckpt", priority=Priority.HIGH)
class ModelCkptCallback(Callback):
    def __init__(
        self,
        prefixfilename: str,
        save_on_epoch: bool=True,
        num_epoch_save: int = 5,
        max_keeps: int = 3,
        keep_condition: str = "val_ssim",
        mode: str = 'max',
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.prefixfilename = Path(prefixfilename)
        self.save_on_epoch = save_on_epoch
        self.num_epoch_save = num_epoch_save
        self.max_keeps = max_keeps
        self.keep_condition = keep_condition
        self.mode = mode

        self.saved_checkpoints = [] 
        

    
    def on_train_end(self, trainer, params = None):
        self._save_checkpoint(
            trainer,
            suffix=f"last_model_epoch_{trainer.state.current_epoch}"
        )
    
    def on_fit_epoch_end(self,
        trainer,
        params: Optional[Dict[str, Any]] = None
    ):

        if trainer.rank == 0:
            global_step = len(trainer.train_loader) * trainer.state.current_epoch 
            if "_" not in self.keep_condition:
                raise ValueError(
                    f"Warning: Invalid monitor format for model checkpoint saving, please indicate with {{metric_type}}_{{metric_name}}, like val_ssim, or train_loss"
                )
                
    
            monitor_type, metric_name = self.keep_condition.split("_",1)
            monitor_type = monitor_type.lower()
            
            if monitor_type == "train":
                if not hasattr(trainer.train_metrics, metric_name):
                    raise ValueError(f"Monitor metric {metric_name} is not valid in train metrics")
                metric_value = getattr(trainer.train_metrics, metric_name).tracker.history[-1]
            elif monitor_type == "val":
                if not hasattr(trainer.val_metrics, metric_name):
                    raise ValueError(f"Monitor metric {metric_name} is not valid in val metrics")
                metric_value = getattr(trainer.val_metrics, metric_name).tracker.history[-1]
            else:
                raise ValueError(f"Monitor type {monitor_type} is not valid")
            
    
            suffix=f"epoch_{trainer.state.current_epoch}_step_{global_step}_{metric_name}_{metric_value}"
            self._save_checkpoint(
                trainer,
                 suffix=suffix,
                metric_value=metric_value,
                global_step=global_step
            )
            
            self._manage_checkpoints()

        trainer.accelerator.wait_for_everyone()
    
    def _save_checkpoint(self, trainer, suffix:str, metric_value: Optional[float] = None, global_step: Optional[int]=None) -> None:
       
        
        save_path = self.prefixfilename / f"{suffix}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        
        state_dict = {
            'epoch': trainer.state.current_epoch,
            'global_step': global_step,
            'model_state_dict': trainer.accelerator.unwrap_model(trainer.model).state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            "metric_value": metric_value if metric_value else 0,
            "metric_name": self.keep_condition
           
        }

        torch.save(state_dict, save_path)

        if metric_value is not None:
            self.saved_checkpoints.append({'global_step': global_step, 'metric': metric_value, 'path': str(save_path)})

    def _manage_checkpoints(self):
        if len(self.saved_checkpoints) <= self.max_keeps:
                return
        
        if self.mode == 'min':
            self.saved_checkpoints.sort(key=lambda x: x['metric'])  
        else:
            self.saved_checkpoints.sort(key=lambda x: x['metric'], reverse=True)
        
        checkpoints_to_remove = self.saved_checkpoints[self.max_keeps:] 

        for checkpoint in checkpoints_to_remove:
            try:
                os.remove(checkpoint['path'])
                print(f"Removing model checkpoint at path : {checkpoint['path']}")
            except Exception as e:
                print(f"Warning, could not delete path : {checkpoint['path']} , err : {e}")

        self.saved_checkpoints = self.saved_checkpoints[:self.max_keeps]





@CallbackFactory.register(name="early_stopping", priority=Priority.NORMAL)
class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        monitor: str,
        patience: int = 20,
        min_delta: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor

    
    def on_val_end(self, trainer, params: Dict[str, Any]=None):
        if trainer.rank == 0:
            if "_" not in self.monitor:
                raise ValueError(
                    f"Warning: Invalid monitor format for early stopping. Please specify metrics in the form: '{{metric_type}}_{{metric_name}}'"
                )
            
            monitor_type, metric_name = self.monitor.split("_",1)
            monitor_type = monitor_type.lower()
    
            if monitor_type == "train":
                if not hasattr(trainer.train_metrics, metric_name):
                    raise ValueError(f"Monitor metric {metric_name} is not valid in train metrics")
                   
                metric_tracker = getattr(trainer.train_metrics, metric_name).tracker
            elif monitor_type == "val":
                if not hasattr(trainer.val_metrics, metric_name):
                    raise ValueError(f"Monitor metric {metric_name} is not valid in val metrics")
                metric_tracker = getattr(trainer.val_metrics, metric_name).tracker
            else:
                raise ValueError(f"Monitor type {monitor_type} is not valid")
    
            should_stop = metric_tracker.should_early_stop(
                    patience = self.patience,
                    min_delta = self.min_delta
            )
    
            if should_stop:
                trainer.state.early_stop = True
                if trainer.rank == 0:
                    trainer.logger.log(f"Early stopping trigger since the metric [{metric_name}] is not improving anymore")
        trainer.accelerator.wait_for_everyone()
        

        
@CallbackFactory.register(name='ema', priority=Priority.NORMAL)
class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.initialized = False
    

    def _get_model(self, trainer):
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module
        return model
    
    def _initialize_shadow_params(self, trainer):
        model = self._get_model(trainer)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().cpu()
        
        self.initialized = True
        if trainer.rank == 0:
            trainer.logger.log("EMA initialized")
    
    def on_train_start(self, trainer, params = None):
        self._initialize_shadow_params(trainer)
    
    
    def after_forward_backward(self, trainer, params: Optional[Dict[str, Any]] = None):
        model = self._get_model(trainer)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not found in EMA shadow dictionary"
                    self.shadow[name] = self.shadow[name].to(param.device)
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
                    self.shadow[name] = self.shadow[name].cpu()
    
    def on_val_batch_start(self, trainer, params = None):
        if not self.initialized:
            return
        
        model = self._get_model(trainer)
        trainer.accelerator.wait_for_everyone()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not found in EMA shadow dictionary"
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device))
        
    def on_val_batch_end(self, trainer, params = None):
        if not self.initialized:
            return
        model = self._get_model(trainer)
        trainer.accelerator.wait_for_everyone()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"Parameter {name} not found in backup dictionary"
                param.data.copy_(self.backup[name].to(param.device))
        self.backup.clear()
        

@CallbackFactory.register(name='swa', priority=Priority.HIGH)
class SWACallback(Callback):
    def __init__(
        self,
        swa_start: int = 10,
        swa_freq: int = 5,
        swa_lr: int = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.swa_start = swa_start 
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.opt = None
    
    def on_pretrain_routine_start(self, trainer, params = None):

        base_opt = trainer.optimizer
        if not isinstance(base_opt, torch.optim.SGD):
            if trainer.rank == 0:
                trainer.logger.log(f"Warning: Only SGD optimizer is allowed to use in couple with SWA")
            
        
        swa_opt  = SWA(
            base_opt,
            swa_start=self.swa_start,
            swa_freq=self.swa_freq,
            swa_lr=self.swa_lr
        )
        setattr(trainer, 'swa_opt', swa_opt)
        

    def after_forward_backward(self, trainer, params = None):
        if not isinstance(trainer.optimizer, torch.optim.SGD):
            return
        trainer.swa_opt.swap_swa_sgd()


@CallbackFactory.register(name='communication_hook_ddp', priority=Priority.NORMAL)
class CommunicationHookCallback(Callback):
    def __init__(
        self,
        type: str = 'fp16',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type
    
    def on_pretrain_routine_start(
        self,
        trainer,
        params: Dict[str, Any] = None
    ):
        
        if not isinstance(trainer.model, DDP):
            return

        if self.type ==  "fp16":
            trainer.model.register_comm_hook(
                state=None,
                hook=default_hooks.fp16_compress_hook
            )
        elif self.type ==  "bf16":
            trainer.model.register_comm_hook(
                state=None,
                hook=default_hooks.bf16_compress_hook
            )
        
        elif self.type ==  "powersgd":
            trainer.model.register_comm_hook(
                state=powerSGD_hook.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=self.config.training.powersgd_rank
                ),
                hook=powerSGD_hook.powerSGD_hook
            )

@CallbackFactory.register(name="timer", priority=Priority.LOWEST)
class TimerCallback(Callback):

    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        self.timing_stats = {
            'total_time': 0,
            'epoch_times': [],
            'batch_times': [],
            'validation_times': [],
            'average_batch_time': 0,
            'estimated_time_remaining': 0
        }

    def on_train_start(
        self, 
        trainer,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.start_time = time.time()
        self._log_timing_start(trainer)
    
    def on_train_epoch_start(self, trainer, params: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_start_time = time.time()
        self.batch_times = []
    
    def on_train_batch_start(self, trainer, params: Optional[Dict[str, Any]] = None) -> None:
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, params: Optional[Dict[str, Any]] = None) -> None:
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(
            batch_time
        )
        self.timing_stats['batch_times'].append(batch_time)

        self.timing_stats['average_batch_time'] = np.mean(self.batch_times).item()

        self._update_time_estimation(trainer)

        if trainer.state.global_step % trainer.config.logging.interval == 0:
            self._log_timing_stats(trainer)
        

    def on_train_epoch_end(
        self,
        trainer, 
        params: Optional[Dict[str, Any]] = None
    ):
        epoch_time = time.time() - self.epoch_start_time
        self.timing_stats['epoch_times'].append(epoch_time)
        self._log_epoch_timing(trainer, epoch_time
        )

    def on_train_end(self, trainer, params: Optional[Dict[str, Any]]=None):
        total_time = time.time() - self.start_time
        self.timing_stats['total_time'] = total_time
        self._log_final_timing(trainer, total_time)
    
    def _update_time_estimation(self, trainer) -> None:
        if not self.batch_times:
            return
        
        avg_batch_time = np.mean(self.batch_times).item()
        total_batches = len(trainer.train_loader) * trainer.config.training.num_epochs
        global_step = len(trainer.train_loader) * trainer.state.current_epoch + trainer.state.current_batch_step
        remaining_batches = total_batches - global_step 

        self.timing_stats['estimated_time_remaining'] = avg_batch_time * remaining_batches



    def _log_timing_start(self, trainer) -> None:
        if trainer.rank == 0:
            trainer.logger.log(
                "Training started at: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}"
            )
        trainer.accelerator.wait_for_everyone()
    
    def _log_timing_stats(self, trainer) -> None:
        global_step = len(trainer.train_loader) * trainer.state.current_epoch + trainer.state.current_batch_step
        if trainer.rank == 0:
            log_str = (
                "\n"
                f"Step: {global_step}, "
                f"Avg Batch Time: {as_minutes(self.timing_stats['average_batch_time'])}, "
                f"Estimated Time Remaining: {as_minutes(self.timing_stats['estimated_time_remaining'])}"
                "\n"
            )
            trainer.logger.log(log_str)
        
        trainer.accelerator.wait_for_everyone()
    
    def _log_epoch_timing(self, trainer, epoch_time: float) -> None:
        if trainer.rank == 0:
            log_str = (
                f"Epoch {trainer.state.current_epoch} completed in: {as_minutes(epoch_time)}"
            )
            trainer.logger.log(log_str)
        trainer.accelerator.wait_for_everyone()
    
    def _log_final_timing(self, trainer, total_time: float) -> None:
        if trainer.rank == 0:
            log_str = (
                f"Training finished in: {as_minutes(total_time)}"
            )
            trainer.logger.log(log_str)
        
        trainer.accelerator.wait_for_everyone()
    
    



    
@CallbackFactory.register(name='awp', priority=Priority.NORMAL)
class AWPCallback(Callback):

    def __init__(
        self,
        adv_param='weight',
        adv_lr=0.001,
        adv_eps=0.001,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.awp_applied  = False # first pertubation is not going to activate
    
    def _get_model(self, trainer):
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module
        return model
    
    def _attack_step(self, trainer, params=None):
        e = 1e-6
        model = self._get_model(trainer)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and trainer.config.callbacks.awp.adv_param in name:
                
                optimizer = getattr(trainer.optimizer, 'optimizer', trainer.optimizer)
                grad = optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())
            
                if norm_grad != 0 and not torch.isnan(norm_grad):
                        limit_eps = trainer.config.callbacks.awp.adv_eps * param.detach().abs()
                        param_min = param.data - limit_eps
                        param_max = param.data + limit_eps
                        param.data.add_(grad, alpha=(trainer.config.callbacks.adv_lr * (norm_data + e) / (norm_grad + e)))
                        param.data.clamp_(param_min, param_max)
                    
    def _save(self, trainer, params) :
        model = self._get_model(trainer)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and trainer.config.callbacks.awp.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)
    
    def after_forward_backward(self, trainer, params=None):
        # restore function
        model = self._get_model(trainer)
        
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
    
    def before_forward_backward(self, trainer, params=None):
        # pertubation
        if trainer.config.callbacks.get("awp", {}).get("enabled", False) and self.awp_applied:
            self._attack_step(trainer, params)
            self._save(trainer, params)
             
    def optimizer_step(self, trainer):
       if trainer.config.callbacks.get("awp", {}).get("enabled", False):
          self.awp_applied = True
    

@CallbackFactory.register(name='gradient_monitor', priority=Priority.NORMAL)
class GradientMonitorCallback(Callback):
    def __init__(self, log_interval: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.log_interval = log_interval
    
    def _get_model(self, trainer):
        model = trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        return model

    def on_train_start(self, trainer, params: Optional[Dict[str, Any]] = None):
        trainer.gradient_norm_history = []
    
    def optimizer_step(self, trainer, params: Optional[Dict[str, Any]] = None):
        if trainer.rank == 0:
            model = self._get_model(trainer)
            global_step = trainer.state.current_epoch * len(trainer.train_loader) + trainer.state.current_batch_step
            if global_step % self.log_interval == 0:
                grad_norms = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms[name] = grad_norm
                    
                
                grad_list = list(grad_norms.values())
                global_grad = (sum(g**2 for g in grad_list) ** 0.5) if grad_list else 0.0
                

                trainer.gradient_norm_history.append({
                    "global_step": global_step,
                    "global_gradient_norm": global_grad
                })

        trainer.accelerator.wait_for_everyone()

@CallbackFactory.register(name="visualization", priority=Priority.LOWEST)
class VisualizationCallback(Callback):
    def __init__(self, output_dir: str = "plots", **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, trainer, params: Optional[Dict[str, Any]] = None):
        self._plot_metrics(trainer)
        self._plot_gradient(trainer)


    
    def _plot_metrics(self, trainer):
        import matplotlib.pyplot as plt
        import seaborn as sns

        train_metrics_history = trainer.train_metrics.get_summary_history() 
        val_metrics_summary = trainer.val_metrics.get_summary_history()

        metric_names = list(train_metrics_history.keys())

        for metric_name in metric_names:
            plt.figure(figsize=(10, 6))
            
            train_data = [
                {'global_step': index, 'current': value} 
                for index, value in enumerate(train_metrics_history[metric_name])
            ]
            val_data = [
                {'global_step': index, 'current': value} 
                for index, value in enumerate(val_metrics_summary[metric_name])
            ]
            sns.lineplot(x=[item['global_step'] for item in train_data], y=[item['current'] for item in train_data], label=f'Train {metric_name}')
            sns.lineplot(x=[item['global_step'] for item in val_data], y=[item['current'] for item in val_data], label=f'Validation {metric_name}')

            plt.title(f'{metric_name} over Global Steps')
            plt.xlabel('Global Step')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            plt.savefig(self.output_dir / f'{metric_name.replace(" ", "_")}_over_steps.png')
            plt.clf()
            plt.close()
    
    def _plot_gradient(self, trainer):
        list_of_global_gradients = [p['global_gradient_norm'] for p in trainer.gradient_norm_history]
        global_steps = [p['global_step'] for p in trainer.gradient_norm_history]
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.lineplot(x=global_steps, y=list_of_global_gradients)
        plt.title(f'Global Gradient Norm for over Global Steps')
        plt.xlabel('Global Step')
        plt.ylabel('Gradient Norm')
        plt.grid(True)
        plt.savefig(self.output_dir / f'global_grad_norm_over_steps.png')
        plt.clf()
        plt.close()

