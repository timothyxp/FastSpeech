from dataclasses import dataclass
from typing import Tuple


@dataclass
class FSConfig:
    vocab_size: int = 10_000
    n_mels: int = 80

    hidden_size: int = 256

    encoder_num_layers: int = 3
    encoder_num_heads: int = 2

    decoder_num_layers: int = 3
    decoder_num_heads: int = 2

    fft_num_filters = 1024
    fft_conv_kernel: Tuple[int] = (3, 3)
    fft_padding: Tuple[int] = (1, 1)

    length_aligner_filter_size: int = 256
    length_aligner_kernel_size: int = 3

    dropout_p: float = 0.1