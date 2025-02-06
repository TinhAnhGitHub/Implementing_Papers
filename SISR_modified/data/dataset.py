import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
import random

import os
from typing import Tuple, List

class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], low_img_size: Tuple[int,int], is_train: True):


        self.low_img_size = low_img_size
        self.images = image_paths
        self.resize = transforms.Resize((low_img_size[0], low_img_size[1]), antialias=True)
        self.is_train = is_train
    
    def __len__(self):
        return len(self.images)

    def normalize(self, input_image: Tensor, target_image: Tensor) -> Tuple[Tensor, Tensor]:
        input_image = input_image * 2 - 1
        target_image = target_image * 2 - 1
        return input_image, target_image

    def random_jitter(self, input_image: Tensor, target_image: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)
        return input_image, target_image

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = transforms.functional.to_tensor(image)
        input_image = self.resize(image).type(torch.float32)
        target_image = image.type(torch.float32)

        input_image, target_image = self.normalize(input_image, target_image)
        if self.is_train:
            input_image, target_image = self.random_jitter(input_image, target_image)
        
        return input_image, target_image


def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:

    data_path = cfg['dataset']['data_path']
    val_size = cfg['dataset']['val_size']
    batch_size = cfg['training']['batch_size']
    low_img_size = cfg['dataset']['low_img_size']
    num_workers = cfg['dataset']['num_workers']


    image_paths = os.listdir(data_path)
    random.shuffle(image_paths)

    train_images = image_paths[:len(image_paths)* (1- val_size)]
    val_images = image_paths[len(image_paths)* (1- val_size):]


    train_dataset = ImageDataset(train_images, low_img_size, True)
    val_dataset = ImageDataset(val_images, low_img_size, data_path, is_train=False)


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

            
