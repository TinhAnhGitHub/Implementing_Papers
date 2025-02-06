from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.loss import loss_factory
from model.ModifiedUnet import ModifiedUnetSR
from optim.optimizer_scheduler import create_optimizer_scheduler
from utils.ema_awp import EMA, AWP
from utils.metrics import ssim, psnr
import numpy  as np
import random


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class SISRModule(pl.LightningModule):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        seed_everything(cfg.get("seed", 42))
        self.cfg = cfg 

        model_cfg = cfg['model']
        self.model = ModifiedUnetSR(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            depth=model_cfg["depth"],
            initial_filters=model_cfg["initial_filters"],
            scale_factor=model_cfg["scale_factor"],
            up_mode=model_cfg["up_mode"],
            use_skip_connections=model_cfg['use_skip_connections']
        )

        loss_cfg = cfg['loss']
        self.criterion = loss_factory(loss_cfg["loss_type"], loss_cfg.get("lambda_g"))

        self.use_ema = cfg["ema"]["use_ema"]
        if self.use_ema:
            self.ema = EMA(self.model, decay=cfg["ema"]["decay"])
            self.ema.register()
        
        self.use_awp = cfg["awp"]["use_awp"]
        self.awp = None
        
        self.awp_start = cfg["awp"].get("awp_start", 1.0)
        self.adv_loss_weight = cfg["awp"].get("adv_loss_weight", 0.1)

     
        self.example_input_array = torch.rand(1, int(model_cfg["in_channels"]), int(cfg['dataset']['low_img_size'][0]),int(cfg['dataset']['low_img_size'][1]))


        self.save_hyperparameters(cfg)

        



    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor :
        
        lr_image, hr_image = batch
        sr_image = self(lr_image)

        loss = self.criterion(sr_image, hr_image)

        ssim_acc = ssim(sr_image, hr_image)
        psnr_db = psnr(sr_image, hr_image)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_psnr", psnr_db, on_step=True, on_epoch=True, prog_bar=True)
        

        if self.use_awp and self.awp is not None and self.current_epoch >= self.awp_start:
            self.awp._save()
            self.awp.perturb()

            adv_pred = self(lr_image)
            adv_loss = self.criterion(adv_pred, hr_image)
            self.awp.restore()
            loss += adv_loss * self.adv_loss_weight
        
        return loss
    
    def training_step_end(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.use_ema:
            self.ema.update()
        return outputs

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        lr_image, hr_image = batch
    
        if self.use_ema:
            self.ema.apply_shadow()
    
        sr_image = self(lr_image)
        loss = self.criterion(sr_image, hr_image)
        ssim_acc = ssim(sr_image, hr_image)
        psnr_acc = psnr(sr_image, hr_image)
    
        
    
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_ssim", ssim_acc, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr_acc, on_epoch=True, prog_bar=True)
    
        if self.use_ema:
            self.ema.restore()
        
        return {"val_loss": loss, "val_ssim": ssim_acc, "val_psnr": psnr_acc}
    
    def configure_optimizers(self):

        optimizer_cfg = self.cfg['optimizer']
        scheduler_cfg = self.cfg['scheduler']
        warmup_cfg = self.cfg['warmup']

        total_steps = self.trainer.estimated_stepping_batches

        optimizer, scheduler = create_optimizer_scheduler(
            model=self.model,
            optimizer_config=optimizer_cfg,
            scheduler_config=scheduler_cfg,
            warmup_config=warmup_cfg,
            total_steps=total_steps
        )
        if self.use_awp:
            self.awp = AWP(
                model=self.model,
                optimizer=optimizer,
                adv_param=self.cfg["awp"]["adv_param"],
                adv_lr=self.cfg["awp"]["adv_lr"],
                adv_eps=self.cfg["awp"]["adv_eps"],
            )


        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    
