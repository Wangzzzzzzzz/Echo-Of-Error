"""
PDF (Periodicity Decoupling Framework) — forecasting-only variant.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# =====================================================================
# Helper modules (inlined from PDF_layers)
# =====================================================================

class _Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def _get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available.')


# =====================================================================
# JVP-compatible BatchNorm
# =====================================================================

class _FunctionalBatchNorm1d(nn.Module):
    """BatchNorm1d compatible with forward-mode AD (jvp)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self._freeze_stats = False
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x: (N, C, L)
        if self.training:
            mean = x.mean(dim=(0, 2))
            var = x.var(dim=(0, 2), unbiased=False)
            if not self._freeze_stats:
                self.running_mean.data.mul_(1 - self.momentum).add_(self.momentum * mean.data)
                self.running_var.data.mul_(1 - self.momentum).add_(self.momentum * var.data)
            x = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        else:
            x = ((x - self.running_mean[None, :, None])
                 / torch.sqrt(self.running_var[None, :, None] + self.eps))
        if self.weight is not None:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x


class _BatchNormChannelLast(nn.Module):
    """Apply _FunctionalBatchNorm1d to inputs of shape (B, N, D), normalising
    over the D feature dimension. Internally transposes to (B, D, N)."""

    def __init__(self, d_model, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = _FunctionalBatchNorm1d(d_model, eps=eps, momentum=momentum,
                                         affine=affine)

    def forward(self, x):
        # x: (B, N, D) -> (B, D, N) -> BN -> (B, N, D)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x


# =====================================================================
# Positional encoding
# =====================================================================

def _sincos_pos_encoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def _positional_encoding(pe, learn_pe, q_len, d_model):
    if pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'sincos':
        W_pos = _sincos_pos_encoding(q_len, d_model, normalize=True)
    elif pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    else:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# =====================================================================
# Attention & Encoder
# =====================================================================

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v, prev=None,
                key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, pos=None):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.pos = pos
        self.P_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.P_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            res_attention=res_attention
        )
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout)
        )

    def forward(self, Q, K=None, V=None, prev=None,
                key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=256,
                 norm='LayerNorm', attn_dropout=0., dropout=0.,
                 activation="gelu", res_attention=False, pos=None):
        super().__init__()
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention, pos=pos
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = _BatchNormChannelLast(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = _BatchNormChannelLast(d_model)

    def forward(self, src, prev=None, key_padding_mask=None, attn_mask=None):
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev,
                                                key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src,
                                        key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)

        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=None,
                 norm='LayerNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, n_layers=1, pos=None):
        super().__init__()
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(
                q_len, d_model, n_heads=n_heads, d_ff=d_ff,
                norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, res_attention=res_attention, pos=pos
            ) for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores,
                                     key_padding_mask=key_padding_mask,
                                     attn_mask=attn_mask)
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask,
                             attn_mask=attn_mask)
        return output


class _TSTiEncoder(nn.Module):
    """Channel-independent Transformer encoder for one period branch."""

    def __init__(self, c_in, patch_num, patch_len,
                 n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0.,
                 act="gelu", res_attention=True, pe='zeros', learn_pe=True):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len

        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = _positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(
            q_len, d_model, n_heads, d_ff=d_ff,
            norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            activation=act, res_attention=res_attention,
            n_layers=n_layers, pos=self.W_pos
        )

    def forward(self, x):
        # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)           # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                      # [bs, nvars, patch_num, d_model]
        u = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = z.reshape(-1, n_vars, z.shape[-2], z.shape[-1])
        z = z.permute(0, 1, 3, 2)           # [bs, nvars, d_model, patch_num]
        return z


# =====================================================================
# PDF core backbone
# =====================================================================

class _PDFHead(nn.Module):
    """Forecast head: concat (or average) period branches → linear."""

    def __init__(self, context_window, num_period, target_window,
                 head_dropout=0., concat=True):
        super().__init__()
        self.concat = concat
        in_features = context_window * (num_period if concat else 1)
        self.linear = nn.Linear(in_features, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x_list):
        if self.concat:
            x = torch.cat(x_list, dim=-1)
        else:
            x = torch.stack(x_list, dim=-1).mean(dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class _PDFBackbone(nn.Module):
    """
    Multi-period decomposition backbone.

    Input:  [bs, nvars, seq_len]
    Output: list of [bs, nvars, context_window]
    """

    def __init__(self, c_in, context_window, target_window,
                 period, patch_len, stride, kernel_list,
                 serial_conv=False, wo_conv=False, add=False,
                 n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='LayerNorm', attn_dropout=0., dropout=0.,
                 act="gelu", res_attention=True, pe='zeros', learn_pe=True,
                 fc_dropout=0.):
        super().__init__()

        self.period_list = period
        self.period_len = [math.ceil(context_window / p) for p in self.period_list]
        self.kernel_list_2d = [(n, patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = [
            (n, m // 2 if stride is None else stride[i])
            for i, (n, m) in enumerate(self.kernel_list_2d)
        ]
        self.dim_list = [k[0] * k[1] for k in self.kernel_list_2d]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list_2d, self.stride_list))
        ]

        # Padding layers per period branch
        self.pad_layer = nn.ModuleList([nn.ModuleList([
            nn.ConstantPad1d((0, p - context_window % p), 0)
            if context_window % p != 0 else nn.Identity(),
            nn.ConstantPad1d((0, k[1] - p % k[1]), 0)
            if p % k[1] != 0 else nn.Identity()
        ]) for p, k in zip(self.period_list, self.kernel_list_2d)])

        # 2D conv embedding per period branch
        self.embedding = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
            nn.Flatten(start_dim=2)
        ) for i, (k, s) in enumerate(zip(self.kernel_list_2d, self.stride_list))])

        # Transformer encoder per period branch
        self.backbone = nn.ModuleList([nn.Sequential(
            _TSTiEncoder(
                c_in, patch_num=token, patch_len=self.dim_list[i],
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, res_attention=res_attention,
                pe=pe, learn_pe=learn_pe
            ),
            nn.Flatten(start_dim=-2),
            nn.Linear(self.tokens_list[i] * d_model, context_window)
            if self.tokens_list[i] * d_model != context_window else nn.Identity()
        ) for i, token in enumerate(self.tokens_list)])

        self.wo_conv = wo_conv
        self.serial_conv = serial_conv

        if not self.wo_conv:
            self.conv = nn.ModuleList([nn.Sequential(
                *[nn.Sequential(
                    nn.Conv1d(n, n, kernel_size=ks, groups=n, padding=ks // 2),
                    nn.SELU()
                ) for ks in kernel_list],
                nn.Dropout(fc_dropout),
                nn.Flatten(start_dim=-2),
            ) for n in self.period_len])

        self.context_window = context_window

    def forward(self, z):
        """z: [bs, nvars, seq_len]  →  list of [bs, nvars, context_window]"""
        res = []
        if self.wo_conv:
            for i, period in enumerate(self.period_list):
                glo = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                glo = self.pad_layer[i][1](glo)
                glo = self.embedding[i](glo.unsqueeze(-3))
                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                glo = self.backbone[i](glo)
                res.append(glo)
        elif self.serial_conv:
            for i, period in enumerate(self.period_list):
                loc = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                loc = self.conv[i](loc).reshape(z.shape[0], z.shape[1], -1)[..., :z.shape[-1]]
                glo = self.pad_layer[i][0](loc).reshape(z.shape[0] * z.shape[1], -1, period)
                glo = self.pad_layer[i][1](glo)
                glo = self.embedding[i](glo.unsqueeze(-3))
                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                glo = self.backbone[i](glo)
                res.append(glo)
        else:
            for i, period in enumerate(self.period_list):
                glo = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                loc = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
                loc = self.conv[i](loc).reshape(z.shape[0], z.shape[1], -1)[..., :z.shape[-1]]
                glo = self.pad_layer[i][1](glo)
                glo = self.embedding[i](glo.unsqueeze(-3))
                glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
                glo = self.backbone[i](glo)
                res.append(glo + loc)
        return res


# =====================================================================
# Top-level forecasting model following _fc conventions
# =====================================================================

class PDFForecast(nn.Module):
    """
    PDF model for direct forecasting.

    forward(x, *args) with x: [B, L, C]
    Returns: [B, pred_len, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.IN = configs.instance_norm

        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        n_layers = configs.e_layers
        dropout = configs.dropout
        fc_dropout = getattr(configs, 'fc_dropout', 0.0)
        head_dropout = getattr(configs, 'head_dropout', 0.0)

        add_mode = getattr(configs, 'add', False)
        wo_conv = getattr(configs, 'wo_conv', False)
        serial_conv = getattr(configs, 'serial_conv', False)

        kernel_list = getattr(configs, 'kernel_list', [3, 7, 9])
        period = getattr(configs, 'period', [24, 12])
        patch_len = getattr(configs, 'patch_len', [16])
        stride = getattr(configs, 'stride', None)

        num_periods = len(period)
        concat = not add_mode

        # --- Core backbone (no RevIN — we handle instance norm ourselves) ---
        self.pdf_backbone = _PDFBackbone(
            c_in=c_in, context_window=context_window, target_window=target_window,
            period=period, patch_len=patch_len, stride=stride,
            kernel_list=kernel_list,
            wo_conv=wo_conv, serial_conv=serial_conv, add=add_mode,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, norm='LayerNorm', attn_dropout=dropout,
            dropout=dropout, act="gelu", res_attention=True,
            pe='zeros', learn_pe=True, fc_dropout=fc_dropout
        )

        # --- Forecast head ---
        self.head = _PDFHead(
            context_window, num_periods, target_window,
            head_dropout=head_dropout, concat=concat
        )

    def forward(self, x, *args):
        # x: [B, L, C]
        if self.IN:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x /= stdev

        z = x.permute(0, 2, 1)              # [B, C, L]
        branch_list = self.pdf_backbone(z)   # list of [B, C, context_window]
        out = self.head(branch_list)         # [B, C, pred_len]
        out = out.permute(0, 2, 1)           # [B, pred_len, C]

        if self.IN:
            out = out * stdev
            out = out + means

        return out
