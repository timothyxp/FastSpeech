
import torch
from torch import nn
import numpy as np


def dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout=None
):
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn.masked_fill_(attn_mask, -np.inf)

    attn = torch.softmax(attn, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    output = torch.bmm(attn, v)

    return output, attn


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)
