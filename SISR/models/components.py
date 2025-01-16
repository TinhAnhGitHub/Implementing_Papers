from typing import List, Optional
import torch
import torch.nn as nn

class ActivationHandler:
    """Handles the creation of activation functions."""

    @staticmethod
    def get_activation(act_type: str) -> nn.Module:
        """Returns the activation function based on the specified type.

        Args:
            act_type (str): Type of activation function.

        Returns:
            nn.Module: Activation function.
        """
        if act_type == "LeakyReLU":
            return nn.LeakyReLU(inplace=True)
        elif act_type == "ReLU":
            return nn.ReLU(inplace=True)
        elif act_type == "Tanh":
            return nn.Tanh()
        elif act_type == "GELU":
            return nn.GELU(approximate='tanh')
        else:
            raise ValueError(f"Activation {act_type} is not supported.")

class NormalizationHandler:
    """Handles the creation of normalization layers."""

    @staticmethod
    def get_normalization(norm_type: str, out_channels: int) -> nn.Module:
        """Returns the normalization layer based on the specified type.

        Args:
            norm_type (str): Type of normalization.
            out_channels (int): Number of output channels.

        Returns:
            nn.Module: Normalization layer.
        """
        if norm_type == "BatchNorm2d":
            return nn.BatchNorm2d(out_channels)
        elif norm_type == "InstanceNorm2d":
            return nn.InstanceNorm2d(out_channels)
        else:
            raise ValueError(f"Normalization {norm_type} is not supported.")

class DropoutHandler:
    """Handles the creation of dropout layers."""

    @staticmethod
    def get_dropout(dropout_probability: float) -> nn.Module:
        """Returns the dropout layer with the specified probability.

        Args:
            dropout_probability (float): The probability of dropout.

        Returns:
            nn.Module: Dropout layer.
        """
        return nn.Dropout(p=dropout_probability)

class ConvLayer(nn.Module):
    """Basic convolutional layer with configurable component ordering.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        use_norm (bool, optional): Whether to use normalization. Defaults to True.
        use_act (bool, optional): Whether to use activation. Defaults to True.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        act_type (str, optional): Type of activation function. Defaults to "LeakyReLU".
        norm_type (str, optional): Type of normalization. Defaults to "BatchNorm2d".
        dropout_probability (float, optional): The probability of dropout. Defaults to 0.2.
        layer_order (List[str], optional): Order of components in the layer. Defaults to ["conv", "norm", "act", "dropout"].  The list should have conv2d at least, the other components are optional
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_norm: bool = True,
        use_act: bool = True,
        use_dropout: bool = False,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        dropout_probability: float = 0.2,
        layer_order: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if layer_order is None:
            layer_order = ["conv", "norm", "act", "dropout"]

        valid_components = {"conv", "norm", "act", "dropout"}
        if not all(component in valid_components for component in layer_order):
            raise ValueError(f"Invalid layer order. Valid components are: {valid_components}")
        
        if len(layer_order) == 1 and layer_order[0] != 'conv':
            raise ValueError(f"If there is only 1 component, make sure it is conv, not {layer_order[0]}")

        component_layers: dict[str, nn.Module] = {}
        
        component_layers["conv"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_norm,  
        )

        if use_norm and "norm" in layer_order:
                
                component_layers["norm"] = NormalizationHandler.get_normalization(norm_type, out_channels if layer_order.index("norm") > layer_order.index("conv") else in_channels)

        if use_act and "act" in layer_order:
            component_layers["act"] = ActivationHandler.get_activation(act_type)
        
        if use_dropout and "dropout" in layer_order:
            component_layers["dropout"] = DropoutHandler.get_dropout(dropout_probability)

        
        
        layers: List[nn.Module] = []
        for component in layer_order:
            if component in component_layers:
                layers.append(component_layers[component])

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional layer

        Args:
            x (torch.Tensor): Input tensor, with shape (batchsize, in_channels, height, width)

        Returns:
            torch.Tensor: Output Tensor, with shape (batchsize, out_Channels, height, width)
        """
        return self.conv(x)
    



class UpsampleHandler(nn.Module):
    """Handles the creation of upsampling layers based on the specified type."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_type: str = "ConvTranspose2d",
        up_mode: str = "bilinear"
    ) -> None:
        """Initializes the upsampling layer based on the specified type.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            up_type (str, optional): Type of upsampling. Defaults to "ConvTranspose2d".
            up_mode (str, optional): Mode for Upsample. Defaults to "bilinear".
            use_norm (bool, optional): Whether to use normalization. Defaults to True.
            use_act (bool, optional): Whether to use activation. Defaults to True.
            pixel_shuffle_factor (int, optional): Factor for PixelShuffle. Defaults to 2.
            unpool_indices (Optional[torch.Tensor], optional): Indices for MaxUnpool2d. Defaults to None.
        """
        super().__init__()
        self.up_type = up_type

        if up_type == "ConvTranspose2d":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_type == "Upsample":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True if up_mode != "nearest" else None),
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    layer_order=['conv']
                )
            )
        else:
            raise ValueError(f"Upsampling type {up_type} is not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the upsampling layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        return self.up(x)