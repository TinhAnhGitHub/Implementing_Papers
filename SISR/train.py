"""Training pipeline for Super-Resolution tasks with DDP optimization hooks."""
from typing import Tuple, Optional,  Dict
from pathlib import Path
import torch
from torch import Tensor

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
from accelerate import Accelerator
from accelerate.utils import  ProjectConfiguration
from dataclasses import dataclass

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    ProgressColumn,
    Task
)
from rich.text import Text

import psutil
import GPUtil


from data import  SuperResolutionDataset
from models import ModelFactory
from utils import CallbackFactory, CallbackManager
from utils import PatchLoss, MetricCollection
from utils import rank0_first, create_optimizer_and_scheduler, setup, seed_everything, Logger, cleanup_processes




class CustomInfoColumn(ProgressColumn):
    def __init__(self, trainer: 'Trainer'):
        self.trainer = trainer
        self.last_lost = ""
        super().__init__()
    
    def render(self, task: Task) -> Text:
        if self.trainer.batch_cur_loss:
            self.last_loss = f"{self.trainer.batch_cur_loss:.4f}"
    
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        gpu_usage = []
        
        for gpu in GPUtil.getGPUs():
            gpu_usage.append(f"{gpu.name}: {gpu.load*100:.2f}%")
        
        gpu_info = ", ".join(gpu_usage)

        info_str = (
            f"Loss: {self.last_loss}, CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%, "
            f"GPU: {gpu_info}"
        )
        return Text(info_str, style="progress.percentage")



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
        

        self.transforms = SuperResolutionDataset(config)
        self.logger = Logger(
            log_dir=config.logging.log_dir,
            rank=self.rank
        )
        self.accelerator =  self._create_accelerator()
        self.callback_manager = self._setup_callbacks()


        self.state = TrainerState()

        self._setup_env()
        self._init_components()
        self._register_callbacks()

    
    def _setup_progress_bar(self) -> Progress:
        return Progress(
            TextColumn(
                "[bold blue]{task.description}[/bold blue]", justify="right"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            CustomInfoColumn(self)
        )
    
    def _create_accelerator(self) -> Accelerator:
        project_config = ProjectConfiguration(
            project_dir=str(Path(self.config.logging.wandb.dir)),
            automatic_checkpoint_naming=True
        )

        return Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with="wandb",
            project_config=project_config,
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
        base_lr = self.config.optimizer.lr * self.accelerator.num_processes

        steps_per_epoch = (len(self.train_loader) + self.data.batch_size.train - 1) // self.data.batch_size.train

        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model.parameters(), self.config, base_lr, steps_per_epoch
        )

        #* Init loss and metric
        self.criterion = PatchLoss(self.config)
        self.train_metrics = MetricCollection(config=self.config, metric_type="train")
        self.val_metrics = MetricCollection(config=self.config, metric_type="val")
        self._setup_ddp()

        
        self.callback_manager.trigger_event("on_pretrain_routine_start", trainer=self)

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

    def _setup_ddp(self):
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])
            self.logger.log("Model wrapped in DDP")

    def _load_data(self) -> None:
        """Initialize data loaders"""
        train_dataset = self._create_dataset(is_train=True)
        val_dataset = self._create_dataset(is_train=False) if self.config.data.paths.val else None

        self.train_loader = self._create_train_loader(train_dataset)
        self.val_loader = self._create_val_loader(val_dataset)

        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )

    def _create_dataset(self, is_train: bool) -> SuperResolutionDataset:
        return SuperResolutionDataset(
            img_dir=self.config.data.paths.train if is_train else self.config.data.paths.val,
            transform=self.transforms.get_train_transforms() if is_train else self.transforms.get_val_transforms(),
            is_train=is_train,
            scale_factor=self.config.model.scale_factor,
            use_feature_loss=self.config.training.loss.use_feature_loss,
            feature_model=self.config.training.loss.feature_model,
            feature_layer=self.config.training.loss.feature_layer
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
        """Create validation data loader"""
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
        self._setup_env()
        self.callback_manager.trigger_event("on_train_start", trainer=self)

        self.progress_bar = self._setup_progress_bar()
        self.progress_bar.start()


        
        for epoch in range(self.state.current_epoch, self.config.training.num_epochs):
            
            self.state.current_epoch = epoch
            self.callback_manager.trigger_event("on_train_epoch_start", trainer=self)
            self.epoch_start = self.progress_bar.add_task(f"Epoch {self.state.current_epoch+1}", total=len(self.train_loader))

            self._train_epoch()

            if self.val_loader and epoch % self.config.validation.interval == 0:
                self._validate_epoch()
            
            if self.state.early_stop:
                break
            
            self.callback_manager.trigger_event("on_train_epoch_end", trainer=self)
            self.progress_bar.remove_task(self.epoch_task)




        self.callback_manager.trigger_event("on_train_end", trainer=self)
    

    def _train_epoch(self) -> None:
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, batch in enumerate(self.train_loader):
            self.state.global_step = batch_idx
            self.callback_manager.trigger_event("on_train_batch_start", trainer=self)

            with self.accelerator.accumulate(self.model):
                loss = self._train_step(batch)
                
                self.batch_cur_loss = loss
                self.callback_manager.trigger_event("on_train_batch_end", trainer=self)
                self.progress_bar.update(self.epoch_task, advance=1)
                self.state.global_step += 1
            
            
            if batch_idx % self.config.logging.interval == 0: 
                self._log_training_metrics()
        self.progress_bar.close()

    
    def _train_step(self, batch: Dict[str, Tensor]):

        
        self.callback_manager.trigger_event('before_forward_backward', trainer=self)
        outputs, loss = self._forward_pass(batch=batch)
        self._backward_pass(loss)

        self.callback_manager.trigger_event(
            'after_forward_backward', trainer=self
        )

        
        metrics = {
            "Loss": loss.item(),
            "PSNR": (outputs.cpu().detach(), batch['hr_image']),
            "SSIM": (outputs.cpu().detach(), batch['hr_image'])
        }
        self.train_metrics.update(metrics)
        return loss
    
    def _forward_pass(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        
        lr_images = batch['lr_image']
        hr_images = batch["hr_image"]

        outputs, output_feat = self.model(lr_images)

        loss = self.criterion(
            outputs=outputs,
            targets=hr_images,
            lr_features=output_feat,  
            hr_features=batch.get("hr_features")   
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
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
        
        if self.config.training.gradient_accumulation_steps > 1:
            loss = loss / self.config.training.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients and self.config.optimizer.grad_clip:
            if self.config.optimizer.grad_clip.get("type", "norm") == "norm":
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip.value
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
        
        with torch.no_grad(), rank0_first():
            for batch_idx, batch in enumerate(self.val_loader):
                self.callback_manager.trigger_event(
                "on_val_batch_start",
                trainer=self
            )
            self._validation_step(batch)
        
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

        outputs, output_feat = self.model(lr_images)

        loss = self.criterion(
            outputs=outputs,
            targets=hr_images,
            lr_features=output_feat,  
            hr_features=batch.get("hr_features")   
        )
        with torch.no_grad():
            metrics = {
                "Loss": loss.item(),
                "PSNR": (outputs, batch['hr_image']),
                "SSIM": (outputs, batch['hr_image'])
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
                log_message += (
                    f"{metric_name}:\n"
                    f"  Current: {metric_values['current']:.4f}\n"
                    f"  Mean: {metric_values.get('mean', 'N/A'):.4f}\n"
                    f"  Std: {metric_values.get('std', 'N/A'):.4f}\n"
                )    
            
            self.logger.log(log_message)



    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    

    


def train_process(rank: int, world_size: int, config: DictConfig):
    try:
        trainer = Trainer(
            config=config,
            rank=rank,
            world_size=world_size
        )
        trainer.train()
    except Exception as e:
        print(f"Error in process: {e}")
        raise e
    finally:
        cleanup_processes()
