import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import torch
import torch.nn as nn
from .components import ConvLayer
from .components import ActivationHandler, NormalizationHandler, UpsampleHandler


class ResidualBlock(nn.Module):
    def __init__(       
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride:int,
        padding:int,
        use_norm:int,
        use_act:int,
        use_dropout:int,
        act_type:str,
        norm_type:str,
        dropout_probability: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers: List[nn.Module] = []


        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'norm', 'act', 'dropout']
            ))
        layers.append(
            ConvLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'norm']
            )
        )

        self.layers = nn.Sequential(*layers)
        self.conv_skip = None
        if self.in_channels != self.out_channels:
            self.conv_skip = ConvLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_act=False,
                use_dropout=False,
                use_norm=False
            )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_output = self.layers(x) 
        if self.in_channels != self.out_channels:
            x = self.conv_skip(x)
        
        final_output = x_output + x
        return final_output
    

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride:int,
        padding:int,
        use_norm:int,
        use_act:int,
        use_dropout:int,
        act_type:str,
        norm_type:str,
        dropout_probability: float = 0.2,
        up_type: str = "Upsample",
        up_mode: str = "bilinear",
    ):
        
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['norm', 'conv', 'act', 'dropout']
            ))
        layers.append(
            ConvLayer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'act', 'dropout']
            )
        )

        
        if use_act:
            layers.append(
                ActivationHandler.get_activation(act_type=act_type)
            )
            
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_probability))

        self.up = UpsampleHandler(
            in_channels=out_channels,
            out_channels=out_channels,
            up_type=up_type,
            up_mode=up_mode,
        )
        self.layers = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
       
        x = self.up(x)
        return x
    

class EncoderLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride:int,
        padding:int,
        use_norm:int,
        use_act:int,
        use_dropout:int,
        act_type:str,
        norm_type:str,
        num_blocks:int,
        dropout_probability: float = 0.2,
        double_last_block:bool=True
    ):
        super().__init__()
        
        self.layers = []

        for _ in range(num_blocks-1):
            self.layers.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_norm=use_norm,
                    use_act=use_act,
                    use_dropout=use_dropout,
                    act_type=act_type,
                    norm_type=norm_type,
                    dropout_probability=dropout_probability
                )
            )
        
        self.layers.append(
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels*2 if double_last_block else out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability
            )
        )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    

class RUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_norm:bool,
        use_act:bool,
        use_dropout:bool,
        act_type: str = "LeakyReLU",
        norm_type: str = "BatchNorm2d",
        up_type: str = "Upsample",
        dropout_probability:float=0.1,
        up_mode: str = "bilinear",
        features_encode: List[int] = [64, 128, 256, 512],
        features_decode: List[int] = [1024, 512, 384, 256, 96],
        downsample_each_output_layer: List[bool] = [True, True, True, False],
        feature_initial: int = 64,
        feature_near_final: int = 99,
        num_block_each_feature: List[int] = [4, 4,6,2],
        double_last_blocks_each_feature: List[bool] = [True, True, True, True]

    ):
        """Default Setting

    Model 1:
        features_encode=[64, 128, 256, 512],
        features_decode=[1024, 512, 384, 256, 96],
        double_last_blocks_each_feature=[True, True, True, True],
    
    Model 2:
        features_encode=[64, 128, 256, 512],
        features_decode=[512, 512, 384, 256, 96]
        double_last_blocks_each_feature=[False, True, True, True],
    
    Note
        - Feature initial must be equal to the first index of the feature encode list
        - The value of double_last_blocks_each_feature determines whether the feature dimension are doubled at each stage of the encoding process:
            - If double_last_blocks_each_feature[i] = True, the feature at the current stage is doubled before moving to the next stage. For example:
                - From 64 -> 128 because double_last_blocks_each_feature[0] = True.
                - From 128 -> 256 because double_last_blocks_each_feature[1] = True.
            - If double_last_blocks_each_feature[i] = False
        - At the final stage, if double_last_blocks_each_feature[-1] = True, the last feature dimension in the encoding process is doubled and used to transition into the decoding stage:
            - From 512 -> 1024 (first index of features_decode) because double_last_blocks_each_feature[-1] = True.
        - The same logic applies for all stages in the list.

        for example
            features_encode=[64, 64, 128, 256],
            features_decode=[512, 100, 384, 256, 96],
            double_last_blocks_each_feature=[False, True, True, True],
        
        from the features_decode[1:] -> choose any channel number you wish to test
        in channel must be 3 for image
        num_classses should be 3, based on the problem. For SISR, it output the high-solution image, so the number of classes is 3
        - You can choose any value for the feature near final
        - num_block for each encode layer, choose many as you wish
        - Each decode layer only has 1 block, as suggested by the author

    """
        super().__init__()
        self.depth_middle = len(features_encode) 
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.downsample_each_output_layer = downsample_each_output_layer

        self.input = ConvLayer(
            in_channels=in_channels,
            out_channels=feature_initial,
            kernel_size=7,
            stride=1,
            padding=3,
            use_norm=use_norm,
            use_act=use_act,
            use_dropout=use_dropout,
            act_type=act_type,
            norm_type=norm_type,
            dropout_probability=dropout_probability,
            layer_order=['conv', 'norm', 'act', 'dropout']
        )

        self.down_pooling = torch.nn.MaxPool2d(2)

        
        for i, (feature, num_block) in enumerate(zip(
            features_encode,
            num_block_each_feature
        )):
           
            layer_cur = EncoderLayers(
                in_channels=feature if i != 0 else feature_initial,
                out_channels=feature,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                num_blocks=num_block,
                dropout_probability=dropout_probability,
                double_last_block=double_last_blocks_each_feature[i]
            )
            

            self.encode.append(
                layer_cur
            )
        

        self.output_last_block_in_encoder = features_encode[-1] * 2 if double_last_blocks_each_feature[-1] else features_encode[-1]


        self.bottle_neck1 = nn.Sequential(
            NormalizationHandler.get_normalization(
                norm_type=norm_type,
                out_channels=self.output_last_block_in_encoder
            ),
            ActivationHandler.get_activation(
                act_type=act_type
            )
        )

        self.bottle_neck2 = nn.Sequential(
            ConvLayer(
                in_channels=self.output_last_block_in_encoder,
                out_channels=self.output_last_block_in_encoder*2,
                kernel_size=3,
                stride=1,
                padding=1,
                use_act=True,
                use_dropout=True,
                use_norm=False,
                act_type=act_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'act']
            )
        )

        self.bottle_neck3 = nn.Sequential(
            ConvLayer(
                in_channels=self.output_last_block_in_encoder*2,
                out_channels=self.output_last_block_in_encoder,
                kernel_size=3,
                stride=1,
                padding=1,
                use_act=True,
                use_dropout=True,
                use_norm=False,
                act_type=act_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'act']
            )
        )

        


        """
        features_encode: List[int] = [64, 128, 256,512],
        features_decode: List[int] = [512, 512,384,256,96],

        512 + 512 -> 512
        512 + 256*2 -> 384
        384+ 128*2 -> 256
        256 + 64*2 -> 96

        """
        for i in range(len(features_decode) - 1):
            # in channel = decode[i] + encode[len(encode) - i - 1] * 2 if it is double last block else encode[len(encode) - i - 1]

            # out channel = decode[i+1]
            encode_channel_index = len(features_encode) - i - 1
            encode_channel = features_encode[encode_channel_index]

            encode_channel = encode_channel * 2 if double_last_blocks_each_feature[encode_channel_index] else encode_channel

            decode_layer_cur : UpBlock = UpBlock(
                in_channels = features_decode[i] + encode_channel,
                out_channels= features_decode[i+1],
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                up_type=up_type,
                up_mode=up_mode  
            )
        
            self.decode.append(decode_layer_cur)
    
        
        self.before_final = nn.Sequential(
            ConvLayer(
                in_channels=features_encode[0] + features_decode[-1],
                out_channels=feature_near_final,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'act', 'dropout']
            ),
            ConvLayer(
                in_channels=feature_near_final,
                out_channels=feature_near_final,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=use_norm,
                use_act=use_act,
                use_dropout=use_dropout,
                act_type=act_type,
                norm_type=norm_type,
                dropout_probability=dropout_probability,
                layer_order=['conv', 'act', 'dropout']
            
            )
        )


        self.output = ConvLayer(
            in_channels=feature_near_final,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            stride=1,
            use_norm=use_norm,
            use_act=use_act,
            use_dropout=use_dropout,
            act_type=act_type,
            norm_type=norm_type,
            dropout_probability=dropout_probability,
            layer_order=['conv']
        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.input(x)
        output_layer_encode_before_downsampling = [x]
        x = self.down_pooling(x)

        output_layer_encodes = [x]


        for i, layer in enumerate(self.encode):

            output_layer_encode = layer(output_layer_encodes[-1])
            output_layer_encode_before_downsampling.append(output_layer_encode)
            if self.downsample_each_output_layer[i]:
                output_layer_encode = self.down_pooling(output_layer_encode) 
            output_layer_encodes.append(output_layer_encode)
        

        output_bottleneck1 = self.bottle_neck1(output_layer_encodes[-1])
        output_bottleneck2 = self.bottle_neck2(output_bottleneck1)
        output_bottleneck3 = self.bottle_neck3(output_bottleneck2)



        output_addition_bottleneck13 = torch.concat([output_bottleneck1, output_bottleneck3], dim=1)

        
        output_decode_layers: List[torch.Tensor] = [] 
        for i, layer in enumerate(self.decode):
            if i == 0:
                output_decode_layer_init = layer(output_addition_bottleneck13)
            else:
                
                input_to_next_dec = torch.cat(
                    [
                        output_layer_encode_before_downsampling[len(output_layer_encode_before_downsampling) - i - 1], 
                        output_decode_layers[-1]
                    ], 
                
                dim=1)
                output_decode_layer_init = layer(input_to_next_dec)
            output_decode_layers.append(output_decode_layer_init)


        output_before_final = self.before_final(
            torch.concat(
                [
                    output_layer_encode_before_downsampling[0],
                    output_decode_layers[-1]
                ], dim=1
            )
        )

        output = self.output(output_before_final)

        return output

    def __str__(self) -> str:
        model_str = "RUNet Architecture:\n"
        model_str += "===================\n"
        model_str += f"Input Layer: {self.input}\n"
        model_str += f"Down Pooling: {self.down_pooling}\n"
        model_str += "Encoder Layers:\n"
        for i, layer in enumerate(self.encode):
            model_str += f"  Encoder Block {i + 1}: {layer}\n"
        model_str += "Bottleneck Layers:\n"
        model_str += f"  Bottleneck 1: {self.bottle_neck1}\n"
        model_str += f"  Bottleneck 2: {self.bottle_neck2}\n"
        model_str += f"  Bottleneck 3: {self.bottle_neck3}\n"
        model_str += f"  Upsample Bottleneck: {self.up_bottleneck3}\n"
        model_str += "Decoder Layers:\n"
        for i, layer in enumerate(self.decode):
            model_str += f"  Decoder Block {i + 1}: {layer}\n"
        model_str += "Final Layers:\n"
        model_str += f"  Before Final: {self.before_final}\n"
        model_str += f"  Output Layer: {self.output}\n"
        return model_str
        
          
