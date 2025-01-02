import os

import time
from typing import Dict, Optional, Union, List, Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm
import mlflow
import wandb

from models import ModelFactory
from data import TextDataset, collate_fn, TextPreprocessor
from utils import (
    setup, seed_everything, cleanup_processes, log_gpu_metrics, AverageMeter, AWP, EMA, as_minutes
)
from utils import calculate_metrics
from utils import Logger



class Trainer:
    """
    Handles the whole training process
    """
    def __init__(
        self,
        config: OmegaConf,
        rank: int,
        world_size: int
    ):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.logger = Logger(
            log_dir="logs",
            rank= rank,
            config=config
        ).log
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            device_placement=True
        )
        

        self.preprocessor: Optional[TextPreprocessor] = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema: Optional[EMA] = None
        self.awp: Optional[AWP] = None
        self.train_dl: Optional[DataLoader] = None
        self.valid_dl: Optional[DataLoader] = None
        self.train_sampler: Optional[DistributedSampler] = None
        self.criterion = None


        self.best_f1 = 0
        self.patience_counter = 0
        self.start_time = 0
        self.current_epoch = 0
        self.train_metrics = {
            'loss': AverageMeter(),
            'precision': AverageMeter(),
            'recall': AverageMeter(),
            'f1': AverageMeter()
        }   

        self.val_metrics = {
            'loss': AverageMeter(),
            'precision': AverageMeter(),
            'recall': AverageMeter(),
            'f1': AverageMeter()
        }

    
    

    def _log_comparative_metrics(self, step: int) :
        if self.rank == 0 or self.world_size == 1:
            comparative_metrics = {
                'loss': {
                    'train': self.train_metrics['loss'].avg,
                    'val': self.val_metrics['loss'].avg
                },
                'precision': {
                    'train': self.train_metrics['precision'].avg,
                    'val': self.val_metrics['precision'].avg
                },
                'recall': {
                    'train': self.train_metrics['recall'].avg,
                    'val': self.val_metrics['recall'].avg
                },
                'f1': {
                    'train': self.train_metrics['f1'].avg,
                    'val': self.val_metrics['f1'].avg
                }
            }
            if self.config.use_mlflow:
                for metric_name, values in comparative_metrics.items():
                    mlflow.log_metrics({
                        f"train_{metric_name}": values['train'],
                        f"val_{metric_name}": values['val']
                    }, step=step)
                
            if self.config.use_wandb:
                for metric_name, values in comparative_metrics.items():
                    wandb.log({
                        f"{metric_name}_comparision": wandb.plot.line_series(
                            xs=[step],
                            ys=[[values['train']], [values['val']]],
                            keys = ['Train', 'Validation'],
                            title=f"{metric_name.capitalize()} Comparision",
                            xname="step"
                        )
                    }, step=step)
                
        metric_str = " | ".join([
            f"{k.upper()}: train={v['train']:.4f}, val={v['val']:.4f}"
            for k, v in comparative_metrics.items()
        ])

        self.logger(f"Step {step} - {metric_str}")


    def setup(self): 
        if self.world_size > 1:
            setup(
                self.rank, self.world_size
            )
        seed_everything(self.config.seed)
        self.logger(f"Environment setup complete!")
    
    def _load_csv(self, csv_file_path: str, input_col_name: str, label_col_name: str) -> Union[List[str], List[int]]:
        """Reading the csv files, and extract the list of texts and list of labels"""
        import pandas as pd
        self.logger("Load csv...")
        df = pd.read_csv(csv_file_path)
        texts = df[input_col_name].to_list()
        labels = df[label_col_name].apply(lambda x: int(x)).to_list()
        self.logger("CSV loaded!")
        return texts, labels
    

    def load_data(self):
        """load and prepare datasets"""

        self.logger("Setting up the dataset loader")

        #reading the text, labels
        text_training, label_training = self._load_csv(
            self.config.dataset.train_path
        )

        text_val, label_val = self._load_csv(
            self.config.dataset.val_path
        )

        train_dataset = TextDataset(
            texts = text_training,
            labels= label_training,
            preprocessor=self.preprocessor
        )
        valid_dataset = TextDataset(
            texts=text_val,
            labels=label_val,
            preprocessor=self.preprocessor
        )

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            self.train_dl = DataLoader(
                train_dataset, 
                batch_size = self.config.train_params.train_bs,
                sampler = self.train_sampler,
                collate_fn=collate_fn,
                num_workers=self.config.train_params.num_workers,
                pin_memory=True
            )
        else:
            self.train_dl = DataLoader(
                train_dataset,
                batch_size=self.config.train_params.train_bs,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=self.config.train_params.num_workers,
                pin_memory=True
            )
        
        self.logger("Setting up the train loader successfully!")
        
        self.valid_dl = DataLoader(
            valid_dataset,
            batch_size=self.config.train_params.valid_bs,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.train_params.num_workers,
            pin_memory=True
        )

        self.logger("Setting up the valid loader succesfully!")
    
    def initialize_model(self):
        """Initialie model, optimizer, and other stuff"""

        self.model = ModelFactory.create_model(
            "transformer_only",
            vocab_size=len(self.preprocessor.vocab),
            **self.config.model
        )
        self.logger("Model initialized!")
        if self.world_size > 1:
            self.model = DDP(
                self.model, device_ids=[self.rank]
            )
            self.logger("Model initialized with DDP")

        self.optimizer = torch.optim.AdamW(
            self.models.parameters(),
            lr = self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay
        )
        self.logger("Optimizer Initialized!")
        

        total_steps = len(self.train_dl) * self.config.train_params.num_epochs
        warmup_steps = int(total_steps * self.config.train_params.warmup_pct)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.optimizer.lr,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps
        )

        self.logger("Optimizer sceduler initialized!")

        if self.config.train_params.use_ema:
            self.ema = EMA(self.model, self.config.train_params.decay_rate)
            self.ema.register()
            self.logger("EMA initialized")
        
        if self.config.awp.use_awp:
            self.awp = AWP(
                self.model,
                self.optimizer,
                adv_lr=self.config.awp.adv_lr,
                adv_eps=self.config.awp.adv_eps
            )
            self.logger("Adversial Weight Pertubation initialized!")
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger("Criterion initialized!")
        self.model, self.optimizer, self.train_dl, self.valid_dl = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.valid_dl)
    
        self.logger("Accelerator Wrapper Initialized!")


    @torch.no_grad
    def evaluate_valid(self) -> Dict[str, Any]:
        """Evaluates the model on validation set

        Returns:
            Dict[str, Any]: return some metrics dict
        """
        if self.rank == 0 or self.world_size == 1:
            self.model.eval()
            
            if self.ema:
                self.ema.apply_shadow()
                            
            all_preds = []
            all_labels = []
            valid_loss = AverageMeter()

            for batch in tqdm(self.valid_dl, desc="Evaluating"):
                
                input_ids, valid_lengths, labels = batch

                outputs = self.model(
                    X= input_ids,
                    valid_lens= valid_lengths
                )
                
                loss = self.criterion(
                    outputs, labels
                )

                valid_loss.update(loss.item())
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
            
            metrics = calculate_metrics(all_labels, all_preds)
            metrics['valid_loss'] = valid_loss.avg

            if self.ema:
                self.ema.restore()
            

            return metrics
        
        if self.world_size > 1:
            dist.barrier() # wait for rank 0 to complete evaluation
        return {}
    
    @torch.no_grad
    def evaluate_train(self) -> Dict[str, Any]:
        """Evaluates the model on validation set

        Returns:
            Dict[str, Any]: return some metrics dict
        """
        if self.rank == 0 or self.world_size == 1:
            self.model.eval()
            
            if self.ema:
                self.ema.apply_shadow()
                            
            all_preds = []
            all_labels = []
            

            for batch in tqdm(self.train_dl, desc="Evaluating"):
                
                input_ids, valid_lengths, labels = batch

                outputs = self.model(
                    X= input_ids,
                    valid_lens= valid_lengths
                )
                
               

                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
            
            metrics = calculate_metrics(all_labels, all_preds)

            if self.ema:
                self.ema.restore()
            
            return metrics
        
        if self.world_size > 1:
            dist.barrier() # wait for rank 0 to complete evaluation
        return {}
    

    def train_one_epoch(self, epoch: int):
        """trains the model for one epoch"""
        self.model.train()
        if self.world_size > 1:
            self.train_sampler.set_epoch(epoch=epoch)
        
        progress_bar = tqdm(
            total=len(self.train_dl),
            desc=f"[EPOCH {epoch+1}/{self.config.train_params.num_epochs}]",
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
        
        

        for step, batch in enumerate(self.train_dl):
            
            input_ids, valid_lengths, labels = batch

            outputs = self.model(
                X= input_ids,
                valid_lens= valid_lengths
            )

            loss = self.criterion(
                outputs, labels
            )
            loss = loss / self.config.train_params.grad_accumulation

            self.accelerator.backward(loss)
        
            if (step + 1) % self.config.train_params.validation_per_step == 0:
                metrics_val = self.evaluate_valid()
                metrics_train = self.evaluate_train()


                if metrics_val:
                    self.val_metrics['loss'].update(metrics_val['valid_loss'])
                    self.val_metrics['precision'].update(metrics_val['precision'])
                    self.val_metrics['recall'].update(metrics_val['recall'])
                    self.val_metrics['f1'].update(metrics_val['f1'])
                
                if metrics_train:
                    self.train_metrics['loss'].update(loss)
                    self.train_metrics['precision'].update(metrics_train['precision'])
                    self.train_metrics['recall'].update(metrics_train['recall'])
                    self.train_metrics['f1'].update(metrics_train['f1'])
                
                self._log_comparative_metrics(
                    step=len(self.train_dl) * epoch + step
                )
                
                progress_bar.set_postfix({
                    'Rank': self.rank,
                    'Train Loss': f"{self.train_metrics['loss'].avg:.4f}",
                    'Val Loss': f"{self.val_metrics['loss'].avg:.4f}",
                    'Train F1': f"{self.train_metrics['f1'].avg:.4f}",
                    'Val F1': f"{self.val_metrics['f1'].avg:.4f}",
                    'LR': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })
                
                    
                if metrics_val and metrics_val.get(
                    "f1_score", 0
                ) > self.best_f1:
                    self.save_checkpoint(epoch,step,metrics_val)
                    self.best_f1 = metrics_val.get('f1_score')
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.train_params.patience:
                    self.logger("Early stopping triggered.")
                    progress_bar.close()
                    return

            if self.awp and epoch >= self.config.awp.awp_trigger_epoch:
                self.awp.attack_backward(
                    {"input_ids": input_ids, "valid_lengths": valid_lengths, "labels": labels}, 
                    self.accelerator
                )
            
            if (step + 1) % self.config.train_params.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.models.parameters(),
                    self.config.optimizer.grad_clip_value
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.ema:
                    self.ema.update()
            
            progress_bar.update(1)
            if (step + 1) % self.config.train_params.print_gpu_stats_each_steps == 0:
                log_gpu_metrics()
            
        
        progress_bar.close()
        


    def save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Saves model checkpoint."""
        if self.rank == 0 or self.world_size == 1:
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics": metrics
            }

            checkpoint_name = (
                f"checkpoint_epoch{epoch+1}_step{step+1}_",
                f"f1_{metrics['f1_score']:.4f}.pt"           
            )
            torch.save(
                checkpoint,
                os.path.join("checkpoints", checkpoint_name)
            )
            self.logger(f"Saved checkpoint: {checkpoint_name}")
        
    
    def train(self): 
        self.setup()
        self.load_data()
        self.initialize_model()

        self.start_time = time.time()

        for epoch in range(
            self.config.train_params.num_epochs
        ):  
            
            self.train_one_epoch(epoch)
        

        self.logger(f"Training completed. Total time: {as_minutes(time.time() - self.start_time)}")

        cleanup_processes()
    


def train_process(rank: int, world_size: int, config: OmegaConf):
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
    





