import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple
import torch
from omegaconf import DictConfig
import torch.nn as nn
import torchvision.models as models
from .transform import SuperResolutionTransform
from sklearn.model_selection import train_test_split


class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        image_files,
        img_dir,
        transform: Optional[callable] = None,
        is_train: bool = True,
        scale_factor: int = 4
    ):
        self.transform = transform
        self.is_train = is_train
        self.scale_factor = scale_factor
        self.image_files = image_files
        self.img_dir = img_dir
        
    

    
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The output shape might be
            (B, 512, H/32, W/32) for vgg
            (B, 2048, h/32, w/32) for resnet50
        """
        img_path = os.path.join(
            self.img_dir,
            self.image_files[index]
        )
        hr_image = Image.open(img_path).convert('RGB')

        lr_size = (
            hr_image.width // self.scale_factor,
            hr_image.height // self.scale_factor
        )
        lr_image = hr_image.resize(lr_size, Image.BICUBIC)

        

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        output = {
            "lr_image": lr_image,
            "hr_image": hr_image
        }


        return output
        


def create_datasets(
    config: DictConfig,
) -> Tuple[SuperResolutionDataset, Optional[SuperResolutionDataset]]:
    
    train_transform = SuperResolutionTransform(config).get_train_transform()
    val_transform = SuperResolutionTransform(config).get_val_transform()

    img_dir = config.data.path

    image_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith(("png", "jpg", "jpeg"))
    ]

    val_size = config.data.val_size or 0

    if val_size > 0:
        train_files, val_files = train_test_split(
            image_files, test_size=val_size, random_state=config.seed
        )
        train_dataset = SuperResolutionDataset(
            image_files=train_files,
            img_dir=img_dir,
            transform=train_transform,
            is_train=True,
            scale_factor=config.data.scale_factor
        )
        val_dataset = SuperResolutionDataset(
            image_files=val_files,
            img_dir=img_dir,
            transform=val_transform,
            is_train=False,
            scale_factor=config.data.scale_factor
        )
    else:
        train_dataset = SuperResolutionDataset(
            image_files=image_files,
            img_dir=img_dir,
            transform=train_transform,
            is_train=True,
            scale_factor=config.data.scale_factor
        )
        val_dataset = None
    
    return train_dataset, val_dataset