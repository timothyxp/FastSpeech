import torch
from torch import nn
import torch.nn.functional as F
from tts.model.config import FSConfig
from .sublayers import dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q = nn.Linear(hidden_size, hidden_size * num_heads)
        self.k = nn.Linear(hidden_size, hidden_size * num_heads)
        self.v = nn.Linear(hidden_size, hidden_size * num_heads)

        self.ln = nn.LayerNorm(hidden_size)
        self.proc = nn.Linear(hidden_size * num_heads, hidden_size)

        self.drop = nn.Dropout(dropout_p)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, seq_len, hidden, = q.size()

        assert self.hidden_size == hidden

        q_proc = self.prepare(q, self.q, batch_size, seq_len)
        k_proc = self.prepare(k, self.k, batch_size, seq_len)
        v_proc = self.prepare(v, self.v, batch_size, seq_len)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)

        output, attn = dot_product_attention(q_proc, k_proc, v_proc, mask, self.drop)

        output = output.reshape(self.num_heads, batch_size, seq_len, hidden) \
            .permute(1, 2, 0, 3) \
            .reshape(batch_size, seq_len, hidden * self.num_heads)

        result = self.ln(self.drop(self.proc(output)) + q)

        return result, attn

    def prepare(self, x, linear, batch_size, seq_len):
        return linear(x) \
            .reshape(batch_size, seq_len, self.num_heads, self.hidden_size) \
            .permute(2, 0, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, self.hidden_size)


class ConvNet(nn.Module):
    def __init__(self, input_shape, hidden_size, kernels, pads, drop_p=0.0):
        super().__init__()
        self.c1 = nn.Conv1d(input_shape, hidden_size, kernel_size=kernels[0], padding=pads[0])
        self.c2 = nn.Conv1d(hidden_size, input_shape, kernel_size=kernels[1], padding=pads[1])

        self.ln = nn.LayerNorm(input_shape)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        out = self.c2(F.gelu(
            self.c1(x.transpose(-1, -2))
        )).transpose(-1, -2)

        return self.ln(self.drop(out) + x)


class FFT(nn.Module):
    def __init__(self, hidden_size, num_heads, num_filters, kernels, pads, drop_p=0):
        super().__init__()

        self.mh_attn = MultiHeadAttention(hidden_size, num_heads, drop_p)
        self.conv_net = ConvNet(hidden_size, num_filters, kernels, pads, drop_p)

    def forward(self, x, mask=None):
        out, attn = self.mh_attn(x, mask=mask)

        out = self.conv_net(x)

        return out, attn

