import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import torch
import torch.nn as nn

from models import  DoubleConv, Down,Up, UNet


# class TestDoubleConv(unittest.TestCase):
#     def test_double_conv(self):
#         model = DoubleConv(
#             in_channels=3, out_channels=64
#         )
#         x = torch.rand(1,3,32,32)
#         output = model(x)
#         self.assertEqual(
#             output.shape, torch.Size([1,64,32,32])
#         )

#     def test_no_normalization(self):
#         model = DoubleConv(
#             in_channels=3,
#             out_channels=64,
#             use_norm=False
#         )
#         x = torch.rand(1, 3, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))
        
#         for layer in model.conv:
#             for sub_layer in layer.conv:
#                 self.assertNotIsInstance(sub_layer, nn.BatchNorm2d, "BatchNorm2d should not be present when use_norm=False")
        
#     def test_custom_layer_order_conv_act_norm(self):
#         model = DoubleConv(
#             in_channels=3,
#             out_channels=64,
#             layer_order=['conv', 'act', 'norm']
#         )
#         x = torch.rand(1,3,32,32)
#         output=model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))
    
#     def test_invalid_activation_type_raises_error(self):
#         with self.assertRaises(ValueError):
#             model = DoubleConv(in_channels=3, out_channels=64, act_type="InvalidActivation")
#             x = torch.rand(1, 3, 32, 32)
#             _ = model(x)
#     def test_invalid_normalization_type_raises_error(self):
    
#         with self.assertRaises(ValueError):
#             model = DoubleConv(in_channels=3, out_channels=64, norm_type="InvalidNorm")
#             x = torch.rand(1, 3, 32, 32)
#             _ = model(x)
    
#     def test_large_input_output_channels(self):
#         model = DoubleConv(in_channels=1024, out_channels=2048)
#         x = torch.rand(1, 1024, 32, 32)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 2048, 32, 32]))


# class TestDownSample(unittest.TestCase):
#     def test_downsample_default(self):
#         model = Down(
#             in_channels=3,
#             out_channels=64
#         )
#         input = torch.rand(1,3,32,32)
#         output = model(input)
#         self.assertEqual(
#             output.shape, torch.Size([1,64,16,16])
#         )
    
#     def test_custom_pooling_type(self):
#         model = Down(in_channels=3, out_channels=64, pool_type="MaxPool2d")
#         x = torch.rand(1, 3, 64, 64)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))

#     def test_no_normalization(self):
#         model = Down(in_channels=3, out_channels=64, use_norm=False)
#         x = torch.rand(1, 3, 64, 64)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))
    
#     def test_large_input_output_channels(self):
#         """Test Down with a large number of input and output channels."""
#         model = Down(in_channels=1024, out_channels=2048)
#         x = torch.rand(1, 1024, 64, 64)
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 2048, 32, 32]))


# class TestUpSample(unittest.TestCase):
#     def test_typical_usage_with_skip(self):
#         model = Up(in_channels=64, out_channels=32)
#         x = torch.rand(1, 64, 32, 32)  
#         skip = torch.rand(1, 32, 64, 64)  
#         output = model(x, skip)
#         self.assertEqual(output.shape, torch.Size([1, 32, 64, 64]))

    
#     def test_typical_usage_without_skip(self):
#         model = Up(in_channels=64, out_channels=32, use_skip=False)
#         x = torch.rand(1, 64, 32, 32)  # Input tensor
#         output = model(x)
#         self.assertEqual(output.shape, torch.Size([1, 32, 64, 64]))

#     def test_upsample_bilinear(self):
#         model = Up(in_channels=64, out_channels=32, up_type="Upsample", up_mode="bilinear")
#         x = torch.rand(1, 64, 32, 32)
#         skip = torch.rand(1, 32, 64, 64)
#         output = model(x, skip)
#         self.assertEqual(output.shape, torch.Size([1, 32, 64, 64]))

#     def test_large_input_output_channels(self):
#         model = Up(in_channels=1024, out_channels=512)
#         x = torch.rand(1, 1024, 32, 32)
#         skip = torch.rand(1, 512, 64, 64)
#         output = model(x, skip)
#         self.assertEqual(output.shape, torch.Size([1, 512, 64, 64]))



class TestUnetArchitecture(unittest.TestCase):
   

    def test_no_skip_connections(self):
        model = UNet(in_channels=3, num_classes=3, use_skip=True)
        x = torch.rand(1, 3, 1024, 1024)
        output, x_feat = model(x)
        self.assertEqual(output.shape, torch.Size([1, 3, 1024, 1024]))
        self.assertEqual(x_feat.shape, torch.Size([1, 512, 2,2]))
if __name__ == "__main__":
    unittest.main()