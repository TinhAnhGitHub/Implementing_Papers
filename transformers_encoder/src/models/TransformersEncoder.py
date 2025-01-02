import math
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn



def masked_softmax(X: Tensor, valid_lens: Optional[Tensor]) -> Tensor:
    """Perform the softmax operation while masking specified elements.

    Args:
        X (Tensor): A 3-d tensor with shape (batchsize, seq_len, d_model) to apply softmax to
        valid_lens (Optional[Tensor]): A 1D or 2D tensor that specifies the lengths of sequences in the batch. Any sequences that is longer than the valid length will have masked elements in the softmax computation

    Returns:
        Tensor: The softmax output with masked values set to 0.
    """

    def _sequence_mask(
        X: Tensor,
        valid_len: Tensor,
        value: float = 0
    ) -> Tensor:
        """Mask the elements of X according to the valid_len by replacing masked elements with "value" value

        Args:
            X (Tensor): A tensor where elements are masked
            valid_len (Tensor): A tensor indicating valid sequence lengths
            value (float, optional): The value to replace masked elements with. Defaults to 0.

        Returns:
            Tensor: The maksed tensor
        """
        maxlen = X.size(1)

        """
        torch.arange(
            maxlen,
            dtype=torch.float32,
            device=X.device
        )[None, :] -> create a range from 0 -> maxlen, then, convert to (1, max_len)

        valid_len[:, None] -> shape(batch, 1)

        the comparision, sort of it will create a (batch, max_len), for each sample, it will be a long 1D tensor of true and false. The true will be with in the range of valid len

        torch.arange(maxlen) will produce:
        tensor([0, 1, 2, 3, 4])

        [None, :] will reshape it to:
        tensor([[0, 1, 2, 3, 4]])  # Shape: (1, 5)

        valid_len[:, None] will reshape valid_len= [3,2] (which has shape (2,)) to:
        tensor([[3], [2]])  # Shape: (2, 1)
        The mask comparison (<):
        tensor([[True, True, True, False, False],  # For the first sequence (valid up to index 3)
                [True, True, False, False, False]]) # For the second sequence (valid up to index 2)
        """
        mask = torch.arange(
            maxlen,
            dtype=torch.float32,
            device=X.device
        )[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1) # choosing the last dimension
    
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(
                valid_lens,
                shape[1] # repeat it for each query in a batch
            )

        else:
            valid_lens = valid_lens.reshape(-1)
        
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_len=valid_lens, value=1e-8)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """Scaled the dot-product attention mechanism, with masking and dropout

    The attention mechanism computes the weighted sum of values based on the similarity between the queries and the keys, with the scaling factor for dot products

    Args:
        dropout (float): The dropout rate to apply to the attention weights.

    forward (queries, keys, values, valid_lens):
            Compute the attention-weighted sum of values.
    """
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) # applied to attention weights
    
    def forward(
        self,
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        valid_lens: torch.Tensor 
    ):
        """
        Perform the forward pass of the scaled dot-product attention.

        Args:
            queries (torch.Tensor): Tensor of shape `(batch_size, num_queries, d)`
                representing the query vectors.
            keys (torch.Tensor): Tensor of shape `(batch_size, num_key_value_pairs, d)`
                representing the key vectors.
            values (torch.Tensor): Tensor of shape `(batch_size, num_key_value_pairs, value_dim)`
                representing the value vectors.
            valid_lens (torch.Tensor, optional): A tensor of shape `(batch_size,)` or 
                `(batch_size, num_queries)` indicating the valid sequence lengths to mask the attention 
                weights. Defaults to `None`.

        Returns:
            torch.Tensor: The weighted sum of the values, shape `(batch_size, num_queries, value_dim)`.
        """

        d = queries.shape[-1]
        """
        If input is a (b \times n \times m) tensor, mat2 is a
        (b \times m \times p) tensor, out will be a
        (b \times n \times p) tensor.
        """
        scores = torch.bmm(
            queries,
            keys.transpose(1,2)
        ) / math.sqrt(d) # shape of scores: (batch_size, num_queries, num_key_value_pairs)

        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(
            self.dropout(self.attention_weights), values
        ) 

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        bias (bool): Whether to include bias in linear layers.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

    def transpose_qkv(
        self,
        X: Tensor
    ) -> Tensor:
        """Transpose for parallel computation of attention heads

        Args:
            X (Tensor): shape (batch_size, seq_len, embed_size)

        Returns:
            Tensor: shape (batch_size * num_heads, seq_len, head_dim)
        """

        X = X.reshape(
            X.shape[0], X.shape[1], self.num_heads, self.head_dim
        )
        X = X.permute(0,2,1,3) # shape( batchsize, num_heads, seq_leb, head_dim)

        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def tranpose_output(self, X:Tensor) -> Tensor:
        """Transpose back after the attention computation

        Args:
            X (Tensor): Shape (batch_size * num_heads, seq_len, head_dim)

        Returns:
            Tensor: Shape (batch_size, seq_len, embed_dim).
        """
        X = X.reshape(-1, self.num_heads, X.shape[1], self.head_dim)
        X = X.permute(0, 2,1,3)
        return X.reshape(X.shape[0], X.shape[1], self.embed_dim)
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_lens: Optional[Tensor] = None
    ) -> Tensor:
        """Compute multi-head attention

        Args:
            queries (Tensor): Shape (batch_size, seq_len, embed_dim).
            keys (Tensor): Shape (batch_size, num_key_value_pairs, embed_dim).
            values (Tensor): Shape (batch_size, num_key_value_pairs, value_dim).
            valid_lens (Optional[Tensor]): Shape (batch_size,) or (batch_size, seq_len).

        Returns:
            Tensor: _description_
        """
        batch_size = queries.shape[0]
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens,
                repeats=self.num_heads,
                dim=0
            )
        attention = DotProductAttention(self.dropout.p)
        
        output = attention(queries, keys, values, valid_lens)
        output = self.tranpose_output(output)
        return self.W_o(output)
    

class PositionWiseFFN(nn.Module):
    """
    Position-wise feed-forward network.

    Args:
        ffn_num_hiddens (int): Number of hidden units.
        ffn_num_outputs (int): Number of output units.
    """

    def __init__(self, input_dim: int, ffn_num_hiddens: int):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, ffn_num_hiddens)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(ffn_num_hiddens, input_dim)

    def forward(self, X:Tensor) -> Tensor:
        """
        Apply position-wise feed-forward network.

        Args:
            X (Tensor): Shape (batch_size, seq_len, ffn_num_hiddens).

        Returns:
            Tensor: Shape (batch_size, seq_len, ffn_num_outputs).
        """
        return self.dense2(self.gelu(self.dense1(X)))
    


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization

    Args:
        norm_shape (tuple): Shape for layer normalization.
        dropout (float): Dropout rate.
    """

    def __init__(self, norm_shape: tuple, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X:Tensor, Y: Tensor) -> Tensor:
        """Apply residual connection and layer normalization

        Args:
            X (Tensor): input tensor 
            Y (Tensor): Output tensor from sub-layer.

        Returns:
            Tensor: Shape same as X.
        """
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block

    Args:
        num_hiddens (int): Embedding dimension.
        ffn_num_hiddens (int): Number of hidden units in FFN.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        use_bias (bool): Whether to include bias in linear layers.
    """

    def __init__(self, num_hiddens: int, ffn_num_hiddens: int, num_heads: int, dropout: float, use_bias: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm((num_hiddens, ), dropout=dropout)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens)
        self.addnorm2 = AddNorm((num_hiddens, ), dropout=dropout)

    def forward(self, X: Tensor, valid_lens: Optional[Tensor] = None) -> Tensor:
        """Apply encoder block

        Args:
            X (Tensor): Shape (batch size, seq_len, num_hiddens)
            valid_lens (Optional[Tensor]): shape (batch size, ) or (batch size, seq len)

        Returns:
            Tensor: shape (batch_size, seq_len, num_hiddens)
        """
        Y = self.attention(X, X, X, valid_lens)
        Y = self.addnorm1(
            X, Y
        )
        Z = self.ffn(Y)
        return self.addnorm2(Y, Z)
    



class TransformerEncoder(nn.Module):
    """
    Transformer encoder.

    Args:
        vocab_size (int): Vocabulary size.
        num_hiddens (int): Embedding dimension.
        ffn_num_hiddens (int): Number of hidden units in FFN.
        num_heads (int): Number of attention heads.
        num_blks (int): Number of encoder blocks.
        dropout (float): Dropout rate.
        use_bias (bool): Whether to include bias in linear layers.
    """
    def __init__(
        self,
        vocab_size: int,
        num_hiddens: int, 
        ffn_num_hiddens: int,
        num_heads: int,
        num_blks: int,
        dropout: float,
        use_bias: bool=False
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens=num_hiddens, dropout=dropout)
        self.blks  = nn.Sequential(
            * [TransformerEncoderBlock(
                num_hiddens=num_hiddens,
                ffn_num_hiddens=ffn_num_hiddens,
                num_heads=num_heads,
                dropout=dropout,
                use_bias=use_bias
            ) for _ in range(num_blks)]
        )
        
    def forward(self, X: Tensor, valid_lens: Optional[Tensor] = None) -> Tensor:
        """Apply transformer encoder

        Args:
            X (Tensor): Input tokens, shape (batch_size, seq_len).
            valid_lens (Optional[Tensor]): Valid sequence lengths, shape (batch_size,) or (batch_size, seq_len).
        Returns:
            Tensor: Encoded representations, shape (batch_size, seq_len, num_hiddens)
        """
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X




class TransformerEncoderCls(nn.Module):
    """Transformer encoder with a classification head for Sentiment Analysis

    Args:
        vocab_size (int) : The size of the vocabulary
        max_length (int) : Maximum sequence length
        num_layers (int) : Number of encoder blocks
        embed_dim (int): Embedding Dimension
        num_heads (int): Number of attention heads.
        ff_dim (int): Hidden dimension of the feed-forward netwrok
        dropout(float): dropout rate
        num_classes(int) : Number of output classes
    """
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: int,
        num_classes: int
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            num_hiddens=embed_dim,
            ffn_num_hiddens=ff_dim,
            num_heads=num_heads,
            num_blks=num_layers,
            dropout=dropout
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_dim, 20)
        self.fc2 = nn.Linear(20, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.max_length = max_length

    def forward(
        self,
        X: torch.Tensor,
        valid_lens: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for the Transformer Encoder with classification head.

        Args:
            X (Tensor): Input tokens, shape (batch_size, seq_len).
            valid_lens (Optional[Tensor]): Valid sequence lengths, shape (batch_size,) or (batch_size, seq_len).

        Returns:
            Tensor: Output logits, shape (batch_size, num_classes).

        This implementation will omit chunking
        """

        X = X[:, :self.max_length]
        encoder_output = self.encoder(X, valid_lens)
        cls_output = encoder_output[:, 0, :]
        fc1_output = self.gelu(self.fc1(cls_output))
        fc1_output = self.dropout(fc1_output)
        logits = self.fc2(fc1_output)
        
        return logits

