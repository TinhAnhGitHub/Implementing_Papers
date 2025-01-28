"""Training pipeline for Super-Resolution tasks with DDP optimization hooks."""
from typing import Tuple, Optional,  Dict
import torch
from torch import Tensor
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
from accelerate import Accelerator
import os
from dataclasses import dataclass
from tqdm.auto import tqdm
import psutil
import GPUtil
import matplotlib.pyplot as plt
from data import  SuperResolutionDataset, SuperResolutionTransform, create_datasets
from models import ModelFactory
from utils import CallbackFactory, CallbackManager
from utils import  MetricCollection
from utils import  create_optimizer_and_scheduler, setup, seed_everything, Logger, cleanup_processes, init_wandb



@dataclass
class TrainerState:
    current_epoch: int = 0
    current_batch_step: int = 0
    early_stop: bool = False

class Trainer:


    def __init__(
        self,
        rank: int,
        world_size: int,
        config: DictConfig,
    ):
          
        self.config = config
        self.world_size = world_size
        self.rank = rank
        

        self.transforms = SuperResolutionTransform(config)
        self.logger = Logger(
            log_dir=config.logging.log_dir,
            rank=self.rank
        )
        self.accelerator =  self._create_accelerator()
        self.callback_manager = self._setup_callbacks()

        self.batch_cur_loss = 0

        self.project_dir = os.path.join(
            config.core.project_dir,
            config.core.run_name
        )

        self.state = TrainerState()

        self._setup_env()
        self._load_data()
        self._init_components()

    
    
    
    def _create_accelerator(self) -> Accelerator:
        return Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision="no" if self.config.core.precision == 'fp32' else self.config.core.precision,
            device_placement=True
        )

    def _setup_callbacks(self) -> CallbackManager:
        return CallbackFactory.create_callback_manager(config=self.config)
    
    def _init_components(self) -> None:
        
        #****** Create models **************************************
        model_architecture = self.config.model.architecture
        
        common_attributes = {
            key: value for key, value in self.config.model.items() 
            if key not in ["runet", "unet", "architecture"]
        }

        specific_attributes = getattr(self.config.model, model_architecture, {})
        merged_attributes = {**common_attributes, **specific_attributes}

        
        self.model = ModelFactory.create_model(
            model_architecture,
            **merged_attributes
        )

        #* Initialize the optimizer and scheduler
        self.model = self.model.to(self.accelerator.device)
        base_lr = self.config.optimizer.lr * self.accelerator.num_processes

        steps_per_epoch = (len(self.train_loader) + self.config.data.batch_size.train - 1) // self.config.data.batch_size.train

        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, self.config, base_lr, steps_per_epoch
        )

        #* Init loss and metric
        self.criterion = torch.nn.MSELoss()
        
        self._setup_ddp()


        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        self.train_metrics = MetricCollection(config=self.config, metric_type="train", device = self.accelerator.device)
        self.val_metrics = MetricCollection(config=self.config, metric_type="val", device = self.accelerator.device)
        self.callback_manager.trigger_event("on_pretrain_routine_start", trainer=self)

    def _setup_ddp(self):
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
            if self.rank == 0:
                self.logger.log("Model wrapped in DDP")

    def _load_data(self) -> None:
        """Initialize data loaders"""
        train_dataset, val_dataset = create_datasets(self.config)

        self.train_loader = self._create_train_loader(train_dataset)
        self.val_loader = self._create_val_loader(val_dataset)

        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )

    def _create_train_loader(self, dataset: SuperResolutionDataset) -> DataLoader:
      
        if self.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
            return DataLoader(
                dataset,
                batch_size=self.config.data.batch_size.train,
                sampler=sampler,
                num_workers=self.config.data.loaders.num_workers,
                pin_memory=self.config.data.loaders.pin_memory,
                persistent_workers=self.config.data.loaders.persistent_workers
            )
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size.train,
            shuffle=True,
            num_workers=self.config.data.loaders.num_workers,
            pin_memory=self.config.data.loaders.pin_memory,
            persistent_workers=self.config.data.loaders.persistent_workers
        )

    def _create_val_loader(self, dataset: Optional[SuperResolutionDataset]) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size.val,
            shuffle=False,
            num_workers=self.config.data.loaders.num_workers,
            pin_memory=self.config.data.loaders.pin_memory
        )

    def _setup_env(self):
        if self.world_size > 1:
            setup(
                self.rank, self.world_size
            )
        seed_everything(self.config.seed)



    def fit(self) -> None:
        self.callback_manager.trigger_event("on_train_start", trainer=self)

        for epoch in range(self.state.current_epoch, self.config.training.num_epochs):
            self.state.current_epoch = epoch 
            self.callback_manager.trigger_event("on_train_epoch_start", trainer=self)
                
            self._train_epoch(epoch)

            if self.val_loader and epoch % self.config.validation.interval == 0:
                self._validate_epoch()
            
            if self.state.early_stop:
                break
    
            self.callback_manager.trigger_event("on_train_epoch_end", trainer=self)

        self.callback_manager.trigger_event("on_train_end", trainer=self)
    

    def _get_process_info(self) -> Dict:
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_usage = []
        if GPUtil.getGPUs():
            for gpu in GPUtil.getGPUs():
                gpu_usage.append(f"{gpu.name}: {gpu.load*100:.2f}%")
            gpu_info = ", ".join(gpu_usage)
        else:
            gpu_info = "No GPU"
        info = {
           "Loss": f"{self.batch_cur_loss:.4f}" if self.batch_cur_loss is not None else "N/A",
            "CPU": f"{cpu_percent:.1f}%",
            "RAM": f"{ram_percent:.1f}%",
            "GPU": gpu_info,
        }
        return info
    
    def _train_epoch(self, epoch:int) -> None:
        self.model.train()
        self.train_metrics.reset()

        total_batches = len(self.train_loader)
        if self.rank == 0:
            with tqdm(
                    total=total_batches,
                    desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
                    unit="batch",
                    leave=True,
            ) as pbar:
              for batch_idx, batch in enumerate(self.train_loader):
                self.state.current_batch_step = batch_idx 
                self.callback_manager.trigger_event("on_train_batch_start", trainer=self)

                with self.accelerator.accumulate(self.model):
                    loss = self._train_step(batch)
                    self.batch_cur_loss = loss
                    self.callback_manager.trigger_event("on_train_batch_end", trainer=self)
                    pbar.set_postfix(
                        self._get_process_info()
                    )
                    pbar.update()

                if batch_idx % self.config.logging.interval == 0:
                    self._log_training_metrics()
        else:
            for batch_idx, batch in enumerate(self.train_loader):
                self.state.current_batch_step = batch_idx 
                self.callback_manager.trigger_event("on_train_batch_start", trainer=self)

                with self.accelerator.accumulate(self.model):
                    loss = self._train_step(batch)
                    self.batch_cur_loss = loss
                    self.callback_manager.trigger_event("on_train_batch_end", trainer=self)

                
    def _train_step(self, batch: Dict[str, Tensor]):

        
        self.callback_manager.trigger_event('before_forward_backward', trainer=self)
        outputs, loss = self._forward_pass(batch=batch)
        self._backward_pass(loss)

        self.callback_manager.trigger_event(
            'after_forward_backward', trainer=self
        )
                
        metrics = {
            "loss": loss.item(),
            "psnr": (outputs, batch['hr_image']),
            "ssim": (outputs, batch['hr_image'])
        }
        self.train_metrics.update(metrics)
        return loss
    
    def _denormalize(self, image):
        image = (image + 1) / 2
        image = image * 255
        return image
    
    def _visualization(
        self, prediction, batch, folder_to_save, is_train
    ):
        os.makedirs(folder_to_save, exist_ok=True)
        global_step =  len(self.train_loader) * self.state.current_epoch + self.state.current_batch_step
        suffix = "train" if is_train else "val"


        file_name = os.path.join(folder_to_save, f'{global_step}_{self.rank}_pred_{suffix}.jpg')
        prediction_denorm = self._denormalize(prediction[0].detach().cpu()).numpy()
        prediction_denorm = np.transpose(prediction_denorm, (1, 2, 0))
        plt.imsave(file_name, prediction_denorm)


        file_name_hr = os.path.join(prediction, f'{global_step}_{self.rank}_hr_{suffix}.jpg')
        hr_denorm = self._denormalize(batch['hr_image'][0].detach().cpu()).numpy()
        hr_denorm = np.transpose(hr_denorm, (1, 2, 0))
        plt.imsave(file_name_hr, hr_denorm)


        file_name_lr = os.path.join(prediction, f'{global_step}_{self.rank}_lr_{suffix}.jpg')
        lr_denorm = self.denormalize(batch['lr_image'][0].detach().cpu()).numpy()
        lr_denorm = np.transpose(lr_denorm, (1, 2, 0))
        plt.imsave(file_name_lr, lr_denorm)


        
    
    def _forward_pass(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        
        lr_images = batch['lr_image']
        hr_images = batch["hr_image"]

        outputs = self.model(lr_images)

        loss = self.criterion(outputs, hr_images)

        
        # print samples
        if self.config.visualize_debug_train:
            print_sample_train = os.path.join(self.project_dir, 'train_samples')
            self._visualization(outputs, batch, print_sample_train, True)        
    
        if torch.isnan(loss) or torch.isinf(loss):
            if self.rank == 0:
                self.logger.log(
                    f"Warning: Loss is {loss.item()}, skipping batch",
                    level="WARNING"
                )
                raise ValueError("Loss is NaN or Inf")
        return outputs, loss
    

    def _log_training_metrics(self):
        if self.accelerator.is_main_process:
            metrics_summary = self.train_metrics.get_summary()
            table_rows = "\n".join(
                f"{k:<20} {v['current']:<15.4f} {v.get('mean', 'N/A'):<15.4f} {v.get('std', 'N/A'):<15.4f}"
                for k, v in metrics_summary.items()
            )
            if self.rank == 0:
                self.logger.log(
                    f"""
                    Training Metrics Summary:
                    {'Metric Name':<20} {'Current Value':<15} {'Mean Value':<15} {'Std Dev':<15}
                    {'-' * 65}
                    {table_rows}
                    """
                )

    def _backward_pass(
        self, loss: Tensor
    ) -> None:
        
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.config.optimizer.grad_clip:
            if self.config.optimizer.grad_clip.get("type", "norm") == "norm":
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip.max_norm,
                    self.config.optimizer.grad_clip.norm_type
                )
            else:
                self.accelerator.clip_grad_value_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip.value
                )
        if not self.accelerator.gradient_accumulation_steps:
            self.optimizer.step()
            self.scheduler.step()  
            self.optimizer.zero_grad(set_to_none=True)
            self.callback_manager.trigger_event("optimizer_step", trainer=self) 
    

    def _validate_epoch(self) -> None:
        self.callback_manager.trigger_event("on_val_start", trainer=self)
        self.model.eval()
        self.val_metrics.reset()
        
        if self.rank == 0:
            total_batches = len(self.val_loader)
            with torch.no_grad():
               with tqdm(
                    total=total_batches,
                    desc=f"Validation {self.state.current_epoch}/{self.config.training.num_epochs}",
                    unit="batch",
                    leave=True,
                ) as pbar:
                    for _, batch in enumerate(self.val_loader):
                        self.callback_manager.trigger_event(
                            "on_val_batch_start",
                            trainer=self
                        )
                        self._validation_step(batch)
                        pbar.update()
        
            self._log_validation_results()
        self.callback_manager.trigger_event("on_val_end", trainer=self)
        self.accelerator.wait_for_everyone()
    
    def _prepare_validation_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            k: v.to(self.accelerator.device) for k, v in batch.items()
        }
    
    def _validation_step(self, batch: Dict[str, Tensor]) -> None:
        self.callback_manager.trigger_event("on_val_batch_start", trainer=self)
        lr_images = batch['lr_image']
        hr_images = batch["hr_image"]

        outputs = self.model(lr_images)

        # sample test
        if self.config.visualize_debug_val:
            print_sample_train = os.path.join(self.project_dir, 'test_samples')
            self._visualization(outputs, batch, print_sample_train, False)

        loss = self.criterion(
            outputs,
            hr_images
        )
        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "psnr": (outputs, batch['hr_image']),
                "ssim": (outputs, batch['hr_image'])
            }
            self.val_metrics.update(metrics)
        
        self.callback_manager.trigger_event("on_val_batch_end", trainer=self)
        

    def _log_validation_results(self) -> None:
        if self.accelerator.is_main_process:
            metrics_summary = self.val_metrics.get_summary()
            
            log_message = (
                f"\nValidation Results - Epoch {self.state.current_epoch}:\n"
                f"{'=' * 50}\n"
                f"\nMetrics:\n"
                f"{'-' * 30}\n"
            )   

            for metric_name, metric_values in metrics_summary.items():
                print(metric_name, metric_values)
                log_message += (
                    f"{metric_name}:\n"
                    f"  Current: {metric_values['current']:.4f}\n"
                    f"  Mean: {metric_values.get('mean', 'N/A'):.4f}\n"
                    f"  Std: {metric_values.get('std', 'N/A'):.4f}\n"
                )    
            if self.rank == 0:

                self.logger.log(log_message)



    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

def train_process(rank: int, world_size: int, config: DictConfig):
    try:

        wandb_config = config.callbacks.wandb

        if config.callbacks.wandb.enabled and rank == 0:
            init_wandb(wandb_config)

        trainer = Trainer(
            config=config,
            rank=rank,
            world_size=world_size
        )
        trainer.fit()
    except Exception as e:
        print(f"Error in process: {e}")
        raise e
    finally:
        cleanup_processes()
