"""
The implementation of Modified Unet architecture: https://arxiv.org/abs/1911.09428

UNetSR: A U-Net based Super-Resolution network with configurable depth and upscale factor,
adapted as a PyTorch Lightning module.
Input:  H x W x C
Output: (scale_factor * H) x (scale_factor * W) x C
"""
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchsummary import summary



class ConvBlock(nn.Module):
    """Fixed conv 3x3 + LeakyReLU with same input size and output size"""
    def __init__(self, in_channels: int, out_channels:int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.conv(x))


class DownBlock(nn.Module):
    """
    Downsampling block for the encoder
    perform 2x2 ConvBlock followed by 2x2 max pooling
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_block(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled


class UpBlock(nn.Module):
    """
    An up-sampling block for the decoder.
    It first upsamples (by a factor of 2) and then concatenates the corresponding skip connection.
    A ConvBlock then fuses the features.
    """
    def __init__(self, in_channels: int, out_channels: int, up_mode: str = 'bilinear',use_skip_connection: bool = True):
        super().__init__()
        
        if up_mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif up_mode == "transposed":
            self.up = nn.ConvTranspose2d(in_channels , in_channels , kernel_size=2, stride=2)
        else:
            raise ValueError("up_mode must be either 'bilinear' or 'transposed'.")
    
        self.conv_block = ConvBlock(in_channels * (2 if use_skip_connection else 1) , out_channels)
        self.use_skip_connection = use_skip_connection
    
    def forward(self, x: torch.Tensor, skip_x: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.up(x)
        
        if  self.use_skip_connection:
            if skip_x is not None:
                diffY = skip_x.size(2) - x.size(2)
                diffX = skip_x.size(3) - x.size(3)

                x = F.pad(
                    x, [diffX //2 , diffX - diffX//2, diffY//2, diffY - diffY//2] # padding [left, right, top bottom]
                )

                x = torch.cat([skip_x, x], dim=1)
            else: 
                raise ValueError("skip_x cannot be None when use_skip_connection is True")

        return self.conv_block(x)

class ExtraUpsample(nn.Module):
    
    def __init__(self, in_channels, up_mode="bilinear"):
        
        super().__init__()

        if up_mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif up_mode == 'transposed':
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"up_mode must be either 'bilinear' or 'transposed', not {up_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)



class ModifiedUnetSR(nn.Module):
    """
    Modified Unet archiecture
    The encoder downsamples the feature maps, the decoder upsamples them,
    and additional upsampling blocks at the end achieve the desired scale factor.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output image.
        depth (int): Number of down/up-sampling blocks.
        initial_filters (int): Number of filters in the first layer.
        scale_factor (int): Overall upscale factor (should be a power of 2).
        up_mode (str): Upsampling mode, either 'bilinear' or 'transposed'.
        use_skip_connections (bool): Whether to use skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        initial_filters: int = 64,
        scale_factor: int = 2,
        up_mode: str = 'bilinear',
        use_skip_connections: bool=True
    ):
        super().__init__()

        if not (scale_factor != 1 and ((scale_factor & (scale_factor - 1)) == 0)):
            raise ValueError("scale_factor should be a power of 2 (e.g., 2, 4, 8, ...).")

        self.depth = depth
        self.scale_factor = scale_factor
        self.up_mode = up_mode
        self.use_skip_connections = use_skip_connections

        self.input_conv = ConvBlock(in_channels=in_channels, out_channels=initial_filters)
        self.down_blocks = nn.ModuleList()


        current_filters = initial_filters
        for _ in range(depth):
            down_block = DownBlock(current_filters, current_filters * 2)
            self.down_blocks.append(down_block)
            current_filters *= 2

        bottleneck_in_channels = current_filters
        self.bottleneck = ConvBlock(bottleneck_in_channels, current_filters)

        self.up_blocks = nn.ModuleList()

        for _ in range(depth):
            up_block = UpBlock(in_channels=current_filters, out_channels=current_filters//2, up_mode=up_mode, use_skip_connection=self.use_skip_connections) 
            self.up_blocks.append(up_block)
            current_filters //=2
        
        self.out_conv = nn.Conv2d(current_filters, out_channels, kernel_size=1)

        num_extra_upsample_block = int(math.log2(scale_factor))

        self.extra_up_samples = nn.ModuleList([
            ExtraUpsample(current_filters, up_mode=up_mode) for _ in range(num_extra_upsample_block)
        ])
    
    def forward(self, x: torch.Tensor):
        
        x = self.input_conv(x)

        if self.use_skip_connections:
          skip_connections = [x]
          for down in self.down_blocks:
              skip, x = down(x)
              skip_connections.append(skip)
        
          x = self.bottleneck(x)
        
          for up in self.up_blocks:
              skip = skip_connections.pop()
              x = up(x, skip) 
        else:
            for down in self.down_blocks:
                _, x = down(x)

            
            x = self.bottleneck(x)
            
            for up in self.up_blocks:
                x = up(x)  
        
        for extra_up in self.extra_up_samples:
            x = extra_up(x)
        
        x = self.out_conv(x)
        return x



if __name__ == "__main__":

    model_with_skip = ModifiedUnetSR(in_channels=3, out_channels=3, depth=4, initial_filters=64, scale_factor=2, up_mode='transposed', use_skip_connections=True)
    print("Model with skip connections:")
    print(summary(model_with_skip, (3, 256, 256)))

    # model_without_skip = ModifiedUnetSR(in_channels=3, out_channels=3, depth=6, initial_filters=64, scale_factor=2, up_mode='transposed', use_skip_connections=False)
    # print("\nModel without skip connections:")
    # print(summary(model_without_skip, (3, 64, 64)))

    


