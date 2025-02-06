import os
import argparse
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from lightning.sisr_module import SISRModule
from data.dataset import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichModelSummary,
    TQDMProgressBar,
    StochasticWeightAveraging,
    Timer
)

def load_config(config_path: str ) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
    


def parse_args():
    parser = argparse.ArgumentParser(description="Train SISR model using PyTorch Lightning.")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--log_name", type=str, default=None, help="Override the WandB run name.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override max epochs from config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None, help="Override gradient accumulation steps.")
    parser.add_argument("--use_swa", action="store_true", help="Enable stochastic weight averaging.")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.accumulate_grad_batches is not None:
        cfg["training"]["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.checkpoint_dir is not None:
        cfg["checkpoint"]["dirpath"] = args.checkpoint_dir

    train_loader, val_loader = get_dataloaders(cfg)

    run_name = args.log_name or cfg["logging"]["run_name"]
    wandb_logger = WandbLogger(
        project=cfg["logging"]["project"],
        name=run_name,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=cfg["checkpoint"]["monitor"],
        mode=cfg["checkpoint"]["mode"],
        save_top_k=cfg["checkpoint"]["save_top_k"],
        dirpath=cfg["checkpoint"]["dirpath"],
        filename="{epoch}-{val_ssim:.2f}-{val_psnr:.2f}",
    )


    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = RichModelSummary(max_depth=2)
    progress_bar = TQDMProgressBar(refresh_rate=30)
    swa_cb = StochasticWeightAveraging(swa_lrs=cfg["training"].get("swa_lrs", 1e-3)) if args.use_swa else None
    timer_cb = Timer()

   
    callbacks = [checkpoint_cb, lr_monitor, model_summary, progress_bar, timer_cb]
    if swa_cb:
        callbacks.append(swa_cb)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accumulate_grad_batches=cfg["training"].get("accumulate_grad_batches", 1),
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    model = SISRModule(cfg)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
    

