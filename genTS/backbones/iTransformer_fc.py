"""
iTransformer — forecasting-only variant.
Reference: https://arxiv.org/abs/2310.06625
"""

import torch
import torch.nn as nn

from external.tslib.layers.Transformer_EncDec import Encoder, EncoderLayer
from external.tslib.layers.SelfAttention_Family import FullAttention, AttentionLayer
from external.tslib.layers.Embed import DataEmbedding_inverted


class iTransformerForecast(nn.Module):
    """
    iTransformer wrapped for the residual-check pipeline.

    forward(x, *args) with x: [B, L, C] -> [B, pred_len, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.IN = configs.instance_norm

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        dropout = configs.dropout
        factor = getattr(configs, 'factor', 1)
        activation = getattr(configs, 'activation', 'gelu')
        output_attention = getattr(configs, 'output_attention', False)

        # Inverted embedding: each variate sequence -> single token of d_model.
        # We pass x_mark=None so embed_type / freq are unused.
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, d_model,
            embed_type='timeF', freq='h',
            dropout=dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Linear projection from d_model to pred_len (per variate token).
        self.projection = nn.Linear(d_model, configs.pred_len, bias=True)

    def forward(self, x, *args):
        # x: [B, L, C]
        if self.IN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdev

        _, _, N = x.shape

        # Embedding: x_mark=None -> pure value embedding.
        enc_out = self.enc_embedding(x, None)        # [B, N, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Project each variate token to its full forecast horizon.
        dec_out = self.projection(enc_out)           # [B, N, pred_len]
        # Slice to enc_in variates (DataEmbedding_inverted does not pad,
        # but kept for parity with the reference implementation).
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        if self.IN:
            dec_out = dec_out * stdev + means

        return dec_out
