import unittest
import torch
import torch.nn as nn

from SISR.models import ResidualBlock, RUNet, UpBlock, EncoderLayers





# class TestResidualBlock(unittest.TestCase):
#     def test_typical_usage(self):
#         model = ResidualBlock(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             dropout_probability=0.1
#         )
#         x = torch.rand(1, 64, 32, 32)  
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))
    
#     def test_input_channels_not_equal_output_channels(self):
#         """Testing skip connection"""
#         model = ResidualBlock(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#         )
#         x = torch.rand(1, 64, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 128, 32, 32]))

#     def test_large_input_output_channels(self):
#         model = ResidualBlock(
#             in_channels=1024,
#             out_channels=1024,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#         )
#         x = torch.rand(1, 1024, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 1024, 32, 32]))


# class TestUpBlock(unittest.TestCase):
#     def test_default_setting(self):
#         model = UpBlock(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#         )
#         x = torch.rand(1,64,32,32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1,64,64,64]))
    
#     def test_custom_upsampling_type(self):
#         """Test UpBlock with custom upsampling type."""
#         model = UpBlock(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             up_type="ConvTranspose2d",
#         )
#         x = torch.rand(1, 64, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 64, 64]))
    

#     def test_custom_upsampling_mode(self):
#         """Test UpBlock with custom upsampling mode."""
#         model = UpBlock(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             up_mode="nearest",
#         )
#         x = torch.rand(1, 64, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 64, 64]))
     
#     def test_large_input_output_channels(self):
#         model = UpBlock(
#             in_channels=1024,
#             out_channels=1024,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#         )
#         x = torch.rand(1, 1024, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 1024, 64, 64]))


# class TestEncodeLayers(unittest.TestCase):
#     def test_typical_usage(self):
#         model = EncoderLayers(
#             in_channels=64,
#             out_channels=64,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_act=True,
#             use_norm=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             num_blocks=5,
#             double_last_block=True
#         )
#         x = torch.rand(1,64,32,32)
#         print(f"Input shape: {x.shape}")
#         for i, layer in enumerate(model.layers):
#             x = layer(x)
#             print(f"Layer {i+1}: outputshape: {x.shape}")
#         self.assertEqual(x.shape, torch.Size([1, 128, 32, 32]))
    
#     def test_no_double_last(self):
#         model = EncoderLayers(
#             in_channels=512,
#             out_channels=512,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_act=True,
#             use_norm=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             num_blocks=2,
#             double_last_block=False
#         )
#         x = torch.rand(1,512,32,32)
#         print(f"Input shape: {x.shape}")
#         for i, layer in enumerate(model.layers):
#             x = layer(x)
#             print(f"Layer {i+1}: outputshape: {x.shape}")
#         self.assertEqual(x.shape, torch.Size([1, 512, 32, 32]))

    
#     def test_large_input_output_channels(self):
        
#         model = EncoderLayers(
#             in_channels=1024,
#             out_channels=1024,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             use_norm=True,
#             use_act=True,
#             use_dropout=False,
#             act_type="LeakyReLU",
#             norm_type="BatchNorm2d",
#             num_blocks=3,
#             double_last_block=True,
#         )
#         x = torch.rand(1, 1024, 32, 32)
#         print(f"Input shape: {x.shape}")
#         for i, layer in enumerate(model.layers):
#             x = layer(x)
#             print(f"Layer {i+1} output shape: {x.shape}")
#         self.assertEqual(x.shape, torch.Size([1, 2048, 32, 32]))

    

class TestRUnet(unittest.TestCase):
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
    def test_runet(self):
        model = RUNet(
            in_channels=3,
            num_classes=3,
            use_norm=True,
            use_act=True,
            use_dropout=False,
            act_type="LeakyReLU",
            norm_type="BatchNorm2d",
            up_type="Upsample",
            dropout_probability=0.1,
            up_mode="bilinear",
            features_encode=[64, 128, 256, 512],
            features_decode=[1024, 512, 384, 256, 96],
            downsample_each_output_layer = [True, True, True, False],
            feature_initial=64,
            feature_near_final=100,
            num_block_each_feature=[4, 4, 6, 2],
            double_last_blocks_each_feature=[True, True, True, True],
        )
        x = torch.rand(1, 3, 256, 256)  
        print("Model structure:")
        print(model)
        output = model(x)
        print(f"Output shape: {output.shape}")
        self.assertEqual(output.shape, torch.Size([1, 3, 256, 256]))
    




if __name__ == "__main__":
    unittest.main()