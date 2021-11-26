import torch
from torch import nn
from tts.model.config import FSConfig
from .layers import FFT
from tts.collate_fn.collate import Batch
from functools import partial
from .sublayers import Transpose


class Encoder(nn.Module):
    def __init__(self, config: FSConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)

        fft_layer = partial(FFT,
            config.hidden_size, config.encoder_num_heads, config.fft_num_filters,
            config.fft_conv_kernel, config.fft_padding
        )

        self.layers = nn.ModuleList([
            fft_layer() for _ in range(config.encoder_num_layers)
        ])

    def forward(self, text_ids):
        x = self.emb(text_ids)

        for layer in self.layers:
            x, _ = layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config: FSConfig):
        super().__init__()

        fft_layer = partial(FFT,
            config.hidden_size, config.decoder_num_heads, config.fft_num_filters,
            config.fft_conv_kernel, config.fft_padding
        )

        self.layers = nn.ModuleList([
            fft_layer() for _ in range(config.decoder_num_layers)
        ])

    def forward(self, x):

        for layer in self.layers:
            x, _ = layer(x)

        return x


class LengthAligner(nn.Module):
    def __init__(self, config: FSConfig):
        super().__init__()

        self.config = config

        self.net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                config.hidden_size, config.length_aligner_filter_size,
                (config.length_aligner_kernel_size,), padding=config.length_aligner_kernel_size // 2
            ),
            Transpose(-1, -2),
            nn.LayerNorm(config.length_aligner_filter_size),
            nn.GELU(),
            nn.Dropout(config.dropout_p),
            Transpose(-1, -2),
            nn.Conv1d(
                config.length_aligner_filter_size, config.length_aligner_filter_size,
                (config.length_aligner_kernel_size,), padding=config.length_aligner_kernel_size // 2
            ),
            Transpose(-1, -2),
            nn.LayerNorm(config.length_aligner_filter_size),
            nn.GELU(),
            nn.Linear(config.length_aligner_filter_size, 1),
            nn.ReLU()
        )

    def forward(self, x, true_durations=None):
        length = self.net(x).squeeze(-1)
        # batch_size, seq_len

        if true_durations is not None:
            np_length = true_durations.exp().cpu().numpy()
        else:
            np_length = length.detach().exp().cpu().numpy()

        np_length = np_length.round().astype(int)

        real_len = np_length.sum(axis=-1)

        align_matrix = torch.zeros((x.shape[0], x.shape[1], int(real_len.max())))
        align_matrix = gen_binary_alignment(align_matrix, np_length)

        x = x.transpose(-1, -2) @ align_matrix
        x = x.transpose(-1, -2)

        return x, length


@torch.no_grad()
def gen_binary_alignment(mask, lengths):
    for i in range(mask.shape[0]):
        cur_length = 0
        for j in range(mask.shape[1]):
            mask[i, j, cur_length: cur_length + lengths[i][j]] = 1
            cur_length += lengths[i][j]

    return mask


"""
a = torch.arange(5).repeat(2, 1) 

b = torch.LongTensor([[
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1]
],[
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]])

a.unsqueeze(1) @ b

tensor([[[0, 1, 2, 3, 4, 4]],

        [[0, 0, 0, 0, 0, 0]]])
"""
