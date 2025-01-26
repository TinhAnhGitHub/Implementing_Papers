import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Tuple
import torch
from omegaconf import DictConfig
import torch.nn as nn
import torchvision.models as models
from .transform import SuperResolutionTransform


class FeatureExtractor(nn.Module):
    def __init__(self, layer_name: str = "features.35"):
        
        super().__init__()
        
       
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.layer_name = layer_name
        self.feature_model = nn.Sequential()

        for name, module in self.model.named_children():
            self.feature_model.add_module(name, module)
            if name == layer_name.split(".")[0]:
                break

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.feature_model(x)

class SuperResolutionDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        img_dir: str,
        transform: Optional[callable] = None,
        is_train: bool = True,
        scale_factor: int = 4,
        use_feature_loss: bool = False,
        feature_model: Optional[str] = "vgg19",
        feature_layer: Optional[str] = "features.35"
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.scale_factor = scale_factor
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.use_feature_loss = use_feature_loss
        if use_feature_loss:
            self.feature_extractor = FeatureExtractor(model_name=feature_model, layer_name=feature_layer)
        else:
            self.feature_extractor = None

    
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

        if self.use_feature_loss and self.feature_extractor:
            with torch.no_grad():
                hr_features = self.feature_extractor(hr_image.unsqueeze(0)).squeeze(0)
                output["hr_features"] = hr_features

        return output
        
