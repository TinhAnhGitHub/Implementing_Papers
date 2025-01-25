from typing import Dict, Any
from .RUnet import RUNet
from .Unet import UNet

from torch import nn as nn


class ModelFactory:
    @staticmethod
    def create_model(model_name: str, **kwargs: Dict[str, Any])-> nn.Module:
        """Factory method to create different model architecture

        Args:
            model_name: Name of the model architecture to create
            **kwargs: Model parameters

        Returns:
            nn.Module: instantiated model
        """
        if model_name.lower() == "unet":
            return UNet(**kwargs)
        elif model_name.lower() == "runet":
            return RUNet(**kwargs)
        else:
            raise ValueError(f"Unknown Model architecture: {model_name}")