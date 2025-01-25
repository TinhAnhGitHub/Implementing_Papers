from typing import List, Optional
import torch
import torch.nn as nn
from .components import ConvLayer, UpsampleHandler
from data import FeatureExtractor

class DoubleConv(nn.Module):
    """Two consecutive convolutional layers with optional activation and normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        use_act (bool, optional): Whether to use activation. Defaults to True.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        act_type (str, optional): Type of activation function. Defaults to "LeakyReLU".
        norm_type (str, optional): Type of normalization. Defaults to "BatchNorm2d".
        dropout_probability (float, optional): The probability of dropout. Defaults to 0.1.
        layer_order (List[str], optional): Order of components in the layer. Defaults to ["conv", "norm", "act", "dropout"].
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        use_act: bool = True,
        use_dropout: bool = False,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        dropout_probability: float = 0.1,
        layer_order: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if layer_order is None:
            layer_order = ["conv", "norm", "act", "dropout"]

        self.conv = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=layer_order,
            ),
            ConvLayer(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=layer_order,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block for the encoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        use_act (bool, optional): Whether to use activation. Defaults to True.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        act_type (str, optional): Type of activation function. Defaults to "LeakyReLU".
        norm_type (str, optional): Type of normalization. Defaults to "BatchNorm2d".
        pool_type (str, optional): Type of pooling. Defaults to "MaxPool2d".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        use_act: bool = True,
        use_dropout: bool = False,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        pool_type: str = "MaxPool2d",
    ) -> None:
        super().__init__()
        if pool_type == "MaxPool2d":
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError(f"Pooling type {pool_type} is not supported.")

        self.double_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            use_act=use_act,
            use_dropout=use_dropout,
            act_type=act_type,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    """Upsampling block for the decoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_skip (bool, optional): Whether to use skip connections. Defaults to True.
        use_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        use_act (bool, optional): Whether to use activation. Defaults to True.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        act_type (str, optional): Type of activation function. Defaults to "LeakyReLU".
        norm_type (str, optional): Type of normalization. Defaults to "BatchNorm2d".
        up_type (str, optional): Type of upsampling. Defaults to "Upsample".
        pixel_shuffle_factor (int, optional): Factor for pixel shuffle. Defaults to 2.
        unpool_indices (torch.Tensor, optional): Indices from MaxPool2d for unpooling. Defaults to None.
        up_mode (str, optional): Upsampling mode if using nn.Upsample. Defaults to "bilinear".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_skip: bool = True,
        use_norm: bool = True,
        use_act: bool = True,
        use_dropout: bool = False,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        up_type: str = "ConvTranspose2d",
        up_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.up_type = up_type


        self.up = UpsampleHandler(
            in_channels=in_channels,
            out_channels=out_channels,
            up_type=self.up_type,
            up_mode=up_mode
        )

        if self.use_skip:
            self.conv = DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
            )
        else:
            self.conv = DoubleConv(
                in_channels=out_channels,
                out_channels=out_channels,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
            )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        
        x = self.up(x)
        if self.use_skip:
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError("Skip connection is not provided.")


        x = self.conv(x)
        return x


class UNet(nn.Module):
    """U-Net architecture for image segmentation.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        depth (int): Depth of the U-Net (number of encoder/decoder levels).
        features (List[int], optional): Number of features at each level. Defaults to [64, 128, 256, 512, 1024].
        use_skip (bool, optional): Whether to use skip connections. Defaults to True.
        use_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        use_act (bool, optional): Whether to use activation. Defaults to True.
        use_dropout (bool, optional): Whether to use dropout. Defaults to False.
        act_type (str, optional): Type of activation function. Defaults to "LeakyReLU".
        norm_type (str, optional): Type of normalization. Defaults to "BatchNorm2d".
        up_type (str, optional): Type of upsampling in decoder. Defaults to "Upsample".
        pool_type (str, optional): Type of pooling in encoder. Defaults to "MaxPool2d".
        pixel_shuffle_factor (int, optional): Factor for pixel shuffle. Defaults to 2.
        up_mode (str, optional): Upsampling mode if using nn.Upsample. Defaults to "bilinear".
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Optional[List[int]] = None,
        use_skip: bool = True,
        use_norm: bool = True,
        use_act: bool = True,
        use_dropout: bool = False,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        up_type: str = "Upsample",
        pool_type: str = "MaxPool2d",
        up_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]
        self.depth = len(features)-1

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.first_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=features[0],
            use_norm=use_norm,
            use_act=use_act,
            use_dropout=use_dropout,
            act_type=act_type,
            norm_type=norm_type,
        )

        for i in range(self.depth):
            self.downs.append(
                Down(
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    use_norm=use_norm,
                    use_act=use_act,
                    use_dropout=use_dropout,
                    act_type=act_type,
                    norm_type=norm_type,
                    pool_type=pool_type,
                )
            )

        self.bottleneck = DoubleConv(
            in_channels=features[self.depth],
            out_channels=features[self.depth],
            use_norm=use_norm,
            use_act=use_act,
            use_dropout=use_dropout,
            act_type=act_type,
            norm_type=norm_type,
        )

        for i in range(self.depth - 1, -1, -1):
            self.ups.append(
                Up(
                    in_channels=features[i + 1],
                    out_channels=features[i],
                    use_skip=use_skip,
                    use_norm=use_norm,
                    use_act=use_act,
                    use_dropout=use_dropout,
                    act_type=act_type,
                    norm_type=norm_type,
                    up_type=up_type,
                    up_mode=up_mode,
                )
            )

        self.final_conv = ConvLayer(
            in_channels=features[0],
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            use_norm=False,
            use_act=False,
        )
        self.feat_ex = FeatureExtractor()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        skip_connections = []
        x = self.first_conv(x)
        print(x.shape)

        for down in self.downs:
            skip_connections.append(x)
            
            x = down(x)
            print(x.shape)

        
        x = self.bottleneck(x)
        print(x.shape)


        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[-(i + 1)] if self.ups[i].use_skip else None)
            print(x.shape)

        x = self.final_conv(x)
        x_feat = self.feat_ex(x.unsqueeze(0)).squeeze(0)
        return x, x_feat