import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm

class TransformerCharLM(nn.Module):
    r"""A transformer model applied for sequence-to-sequence transform. 
        User is able to modified the attributes as needed.
    Args:
        src_vocab: the number of vocabularies in the source sequence (required). 
        tgt_vocab: the number of vocabularies in the target sequence (required). 
        d_model: the dimension of the encoder/decoder embedding models (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    """

    def __init__(self, vocab, d_model=256, n_heads=4, n_encoder_layers=2, d_ff=128, dropout=0.1):
        super(TransformerCharLM, self).__init__()

        self.src_embed = nn.Embedding(vocab, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = EncoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads)
        self.encoder = TransformerEncoder(encoder_layer, n_encoder_layers)
        self.decoder = nn.Linear(d_model, vocab)

        self.vocab = vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self._init_parameters()

    def forward(self, src, src_mask=None):
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the mask for the src sequence (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the encoder output (optional).
        Shape:
            src: [source sequence length, batch size]
            tgt: [target sequence length, batch size]
            src_mask: [source sequence length, source sequence length]
            tgt_mask: [target sequence length, target sequence length]
            memory_mask: [target sequence length, source sequence length]
            Note: The maksed positions are filled with float('-inf').
                  Unmasked positions are filled with float(0.0). Masks ensure that the predictions
                  for position i depend only on the information before position i.
            output: [target sequence length, batch size, tgt_vocab]
            Note: Due to the multi-head attention architecture in the transformer model,
                  the output sequence length of a transformer is same as the input sequence
                  (i.e. target) length of the decode.
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        src = self.src_embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        memory = self.encoder(src, src_mask)
        output = self.decoder(memory)

        return output

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required). 
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        # self.norm = norm

    def forward(self, src, mask=None):
        r"""Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for layer in self.layers:
            output = layer(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x):
        att = self.attn_head(x, x, x)
        x = x + self.dropout(self.layer_norm1(att))
        ff = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(ff))
        return x


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens 
        in the sequence. The positional encodings have the same dimension as 
        the embeddings, so that the two can be summed. Here, we use sine and cosine 
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]    
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        self.attention = ScaledDotProductAttention(dropout)
        self.query_linear = nn.Linear(d_model, d_feature)
        self.key_linear = nn.Linear(d_model, d_feature)
        self.value_linear = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values):
        # Q, K, V  dim = (batch_size, seq_len, d_model)
        Q = self.query_linear(queries)
        K = self.key_linear(keys)
        V = self.value_linear(values)
        x = self.attention(Q, K, V)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        d_k = K.size(-1)
        assert Q.size(-1) == d_k
        # scores dim = (batch_size, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(1, 2))
        scores = scores / math.sqrt(d_k)
        attention = F.softmax(scores, dim = -1)
        attention = self.dropout(attention)
        # output dim = (batch_size, seq_len, d_model)
        output = torch.matmul(attention, V)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_feature = d_model // n_heads
        assert d_model == d_feature * n_heads

        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])

        self.linear_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        x = [attn(queries, keys, values) for _, attn in enumerate(self.attn_heads)]       
        # concatenate heads
        # x dim = (batch_size, seq_len, d_model= d_feature*heads)
        x = torch.cat(x, dim=2)
        x = self.linear_projection(x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x