import unittest
import torch
import math
from torch import Tensor
import torch.nn as nn

from Transfomers.src.models import (
    DotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderBlock,
    TransformerEncoderCls
)

class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 30
        self.seq_length = 50
        self.embed_dim = 64
        self.num_heads = 8
        self.vocab_size = 1000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_positional_encoding(self):
        """Test the positional encoding component"""
        pos_encoder = PositionalEncoding(
            embed_dim=self.embed_dim,
            dropout=0.1,
            max_len=100
        )
        
        # Test shape
        x = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
        encoded = pos_encoder(x)
        self.assertEqual(encoded.shape, x.shape)
        

    def test_dot_product_attention(self):
        """Test the dot product attention mechanism"""
        attention = DotProductAttention(dropout=0.1)
        
        # Create test inputs
        queries = torch.randn(self.batch_size, 10, self.embed_dim)
        keys = torch.randn(self.batch_size, 15, self.embed_dim)
        values = torch.randn(self.batch_size, 15, self.embed_dim)
        valid_lens = torch.tensor([10, 8, 12] * (self.batch_size // 3))
        
        # Test output shape
        output = attention(queries, keys, values, valid_lens)
        self.assertEqual(output.shape, (self.batch_size, 10, self.embed_dim))
        
        # Test attention weights shape
        self.assertEqual(
            attention.attention_weights.shape,
            (self.batch_size, 10, 15)
        )
        

    def test_multi_head_attention(self):
        """Test the multi-head attention mechanism"""
        mha = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Test with same input for Q, K, V (self-attention)
        x = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
        output = mha(x, x, x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Test with different sequence lengths for K, V
        kv_length = 30
        k = torch.randn(self.batch_size, kv_length, self.embed_dim)
        v = torch.randn(self.batch_size, kv_length, self.embed_dim)
        output = mha(x, k, v)
        
        # Check output shape maintains query sequence length
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.embed_dim)
        )

    def test_transformer_encoder_block(self):
        """Test a single transformer encoder block"""
        encoder_block = TransformerEncoderBlock(
            num_hiddens=self.embed_dim,
            ffn_num_hiddens=self.embed_dim * 4,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
        output = encoder_block(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Test with valid_lens
        valid_lens = torch.tensor([40, 35, 45] * (self.batch_size // 3))
        output = encoder_block(x, valid_lens)
        self.assertEqual(output.shape, x.shape)

    def test_transformer_encoder(self):
        """Test the complete transformer encoder"""
        encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            num_hiddens=self.embed_dim,
            ffn_num_hiddens=self.embed_dim * 4,
            num_heads=self.num_heads,
            num_blks=6,
            dropout=0.1
        )
        
        # Test with integer input (token ids)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output = encoder(x)
        
        # Check output shape
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.embed_dim)
        )
        
        # Test with shorter sequence
        x_short = torch.randint(0, self.vocab_size, (self.batch_size, 20))
        output_short = encoder(x_short)
        self.assertEqual(
            output_short.shape,
            (self.batch_size, 20, self.embed_dim)
        )

    def test_transformer_encoder_cls(self):
        """Test the transformer encoder with classification head"""
        num_classes = 5
        model = TransformerEncoderCls(
            vocab_size=self.vocab_size,
            max_length=self.seq_length,
            num_layers=6,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            dropout=0.1,
            num_classes=num_classes
        )
        
        # Test with regular input
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, num_classes))
        
        # Test with different batch sizes
        small_batch = torch.randint(0, self.vocab_size, (5, self.seq_length))
        output_small = model(small_batch)
        self.assertEqual(output_small.shape, (5, num_classes))
        
        # Test with valid_lens
        valid_lens = torch.tensor([40, 35, 45] * (self.batch_size // 3))
        output_masked = model(x, valid_lens)
        self.assertEqual(output_masked.shape, (self.batch_size, num_classes))

    def test_edge_cases(self):
        """Test edge cases and potential error conditions"""
        model = TransformerEncoderCls(
            vocab_size=self.vocab_size,
            max_length=self.seq_length,
            num_layers=6,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            dropout=0.1,
            num_classes=5
        )
        
        # Test with minimum sequence length
        x_min = torch.randint(0, self.vocab_size, (self.batch_size, 1))
        output_min = model(x_min)
        self.assertEqual(output_min.shape, (self.batch_size, 5))
        
        # Test with batch size of 1
        x_single = torch.randint(0, self.vocab_size, (1, self.seq_length))
        output_single = model(x_single)
        self.assertEqual(output_single.shape, (1,5))
        

        x_batch = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))

        
        model_no_dropout = TransformerEncoderCls(
            vocab_size=self.vocab_size,
            max_length=self.seq_length,
            num_layers=6,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            dropout=0.0,
            num_classes=5
        )
        output_no_dropout = model_no_dropout(x_batch)
        self.assertEqual(output_no_dropout.shape, (self.batch_size, 5))

if __name__ == '__main__':
    unittest.main()