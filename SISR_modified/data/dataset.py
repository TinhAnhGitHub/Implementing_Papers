import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import random

import os
from typing import Tuple, List

class ImageDataset(Dataset):
    def __init__(self, img_dir:str, image_paths: List[str], low_img_size: Tuple[int,int], scale_factor: int, is_train: bool = True):
        self.low_img_size = low_img_size
        self.images = image_paths
        
        self.is_train = is_train
        self.img_dir = img_dir
        self.scale_factor = scale_factor

        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  
        ])

        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        

        
    def __len__(self):
        return len(self.images)

    def normalize(self, image: Tensor) -> Tensor:
        image = image * 2 - 1
        return image
    



    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")  

        if self.is_train:
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)

        input_image = F.interpolate(image.unsqueeze(0), size=self.low_img_size, mode='bicubic', antialias=True).squeeze(0)
        target_image = F.interpolate(image.unsqueeze(0), size=(self.low_img_size[0] * self.scale_factor, self.low_img_size[1] * self.scale_factor), mode='bicubic', antialias=True).squeeze(0)


        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                input_image, output_size=(self.low_img_size[0] // 2, self.low_img_size[1] // 2) 
            )
            input_image = transforms.functional.crop(input_image, i, j, h, w)
            target_image = transforms.functional.crop(target_image, i*self.scale_factor, j*self.scale_factor, h*self.scale_factor, w*self.scale_factor)

            input_image = input_image + torch.randn_like(input_image) * 0.05 

        input_image = self.normalize(input_image)
        target_image = self.normalize(target_image)

        return input_image, target_image


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:

    data_path = cfg['dataset']['data_path']
    val_size = cfg['dataset']['val_size']
    batch_size = cfg['training']['batch_size']
    low_img_size = cfg['dataset']['low_img_size']
    num_workers = cfg['dataset']['num_workers']

    scale_factor = cfg['model']['scale_factor']


    image_paths = os.listdir(data_path)
    
    random.shuffle(image_paths)

    train_size = int(len(image_paths) * (1 - val_size)) 
    train_images = image_paths[:train_size] 
    val_images = image_paths[train_size:]


    train_dataset = ImageDataset(data_path, train_images, low_img_size, scale_factor, True)
    val_dataset = ImageDataset(data_path, val_images, low_img_size, scale_factor, False)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

            
