from SISR.models import UpsampleHandler, ActivationHandler, ConvLayer, NormalizationHandler,DropoutHandler, DoubleConv


import unittest
import torch
import torch.nn as nn

class TestActivationHandler(unittest.TestCase):
    def test_get_activation(self):
        act = ActivationHandler.get_activation("LeakyReLU")
        self.assertIsInstance(act, nn.LeakyReLU)
    
        act = ActivationHandler.get_activation('ReLU')
        self.assertIsInstance(act, nn.ReLU)
        
        act = ActivationHandler.get_activation("Tanh")
        self.assertIsInstance(act, nn.Tanh)

        act = ActivationHandler.get_activation("GELU")
        self.assertIsInstance(act, nn.GELU)

        with self.assertRaises(ValueError):
            ActivationHandler.get_activation("UnsupportedActivation")

class TestNormalizationHandler(unittest.TestCase):
    def test_get_normalization(self):
        
        norm = NormalizationHandler.get_normalization("BatchNorm2d", 64)
        self.assertIsInstance(norm, nn.BatchNorm2d)

        norm = NormalizationHandler.get_normalization("InstanceNorm2d", 64)
        self.assertIsInstance(norm, nn.InstanceNorm2d)

        with self.assertRaises(ValueError):
            NormalizationHandler.get_normalization("UnsupportedNorm", 64)


class TestDropoutHandler(unittest.TestCase):
    def test_get_dropout(self):
        dropout = DropoutHandler.get_dropout(0.5)
        self.assertIsInstance(dropout, nn.Dropout)
        self.assertEqual(dropout.p, 0.5)


class TestConvLayer(unittest.TestCase):
    def test_conv_layer_creation(self):
        # default config
        conv_layer = ConvLayer(64, 128, kernel_size=3, stride=1, padding=1)
        self.assertIsInstance(conv_layer.conv[0], nn.Conv2d)
        self.assertIsInstance(conv_layer.conv[1], nn.BatchNorm2d)
        self.assertIsInstance(conv_layer.conv[2], nn.LeakyReLU)

        # Test without norm and act
        conv_layer = ConvLayer(64, 128, kernel_size=3, stride=1, padding=1, use_norm=False, use_act=False)
        self.assertEqual(len(conv_layer.conv), 1)
        self.assertIsInstance(conv_layer.conv[0], nn.Conv2d)

        # dropout
        conv_layer = ConvLayer(64, 128, kernel_size=3, stride=1, padding=1, use_dropout=True)
        self.assertIsInstance(conv_layer.conv[-1], nn.Dropout)

        with self.assertRaises(ValueError):
            ConvLayer(64, 128, kernel_size=3, stride=1, padding=1, layer_order=["invalid"])

    def test_conv_layer_forward(self):
        conv_layer = ConvLayer(
            64,128,kernel_size=3, stride=1,padding=1, use_dropout=True
        )
        x = torch.rand(1,64,32,32)
        output = conv_layer(x)
        self.assertEqual(output.shape, torch.Size([1, 128, 32, 32]))
        
    
class TestUpSampler(unittest.TestCase):
    def test_upsample_handler_creation(self):
        upsample = UpsampleHandler(64, 128, up_type="ConvTranspose2d")
        self.assertIsInstance(upsample.up, nn.ConvTranspose2d)
    
        upsample = UpsampleHandler(64, 128, up_type="Upsample", up_mode="bilinear")
        self.assertIsInstance(upsample.up, nn.Upsample)

        upsample = UpsampleHandler(64, 128, up_type="PixelShuffle", pixel_shuffle_factor=2)
        self.assertIsInstance(upsample.up, nn.PixelShuffle)

        unpool_indices = torch.randint(0, 4, (1, 64, 32, 32))
        upsample = UpsampleHandler(64, 128, up_type="MaxUnpool2d", unpool_indices=unpool_indices)

        self.assertIsInstance(upsample.up, nn.MaxUnpool2d)


        with self.assertRaises(ValueError):
            UpsampleHandler(64, 128, up_type="UnsupportedType")


    def test_upsample_handler_forward(self):

        upsample = UpsampleHandler(64, 128, up_type="ConvTranspose2d")
        x = torch.randn(1, 64, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, torch.Size([1, 128, 64, 64]))
         
        upsample = UpsampleHandler(64, 128, up_type="Upsample", up_mode="bilinear")
        x = torch.randn(1, 64, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, torch.Size([1, 64, 64, 64]))
         
        upsample = UpsampleHandler(64, 128, up_type="PixelShuffle", pixel_shuffle_factor=2)
        x = torch.randn(1, 64, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, torch.Size([1, 16, 64, 64]))
         
        unpool_indices = torch.randint(0, 4, (1, 64, 32, 32))
        upsample = UpsampleHandler(64, 128, up_type="MaxUnpool2d", unpool_indices=unpool_indices)
        x = torch.randn(1, 64, 32, 32)
        output = upsample(x)
        self.assertEqual(output.shape, torch.Size([1, 64, 64, 64]))


if __name__ == "__main__":
    unittest.main()