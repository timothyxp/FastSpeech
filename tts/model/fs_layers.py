import torch
from torch import nn
from tts.model.config import FSConfig
from .layers import FFT
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

    def forward(self, x):
        length = self.net(x).squeeze()
        # batch_size, seq_len

        np_length = length.detach().exp().cpu().numpy()
        real_len = np_length.round()
        align_matrix = torch.zeros((x.shape[0], int(real_len.max())))

        return x, length


def gen_binary_alignment(mask, lengths):
    return mask
