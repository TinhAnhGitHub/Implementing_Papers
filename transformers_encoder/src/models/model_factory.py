from typing import Dict, Any

from Transfomers.src.models import TransformerEncoderCls
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
        if model_name.lower() == "transformer_encoder_only":
            return TransformerEncoderCls(**kwargs)

        else:
            raise ValueError(f"Unknown Model architecture: {model_name}")
    


