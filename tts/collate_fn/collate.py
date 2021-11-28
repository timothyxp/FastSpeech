from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    waveform: torch.Tensor
    waveforn_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    duration_prediction: Optional[torch.Tensor] = None
    melspec: Optional[torch.Tensor] = None
    melspec_prediction: Optional[torch.Tensor] = None

    def to(self, device: torch.device, non_blocking=False) -> 'Batch':
        self.waveform = self.waveform.to(device, non_blocking=non_blocking)
        self.tokens = self.tokens.to(device, non_blocking=non_blocking)

        return self


class LJSpeechCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)


class TestCollator:
    def __call__(self, instances: List[Tuple]) -> Batch:
        tokens, token_lengths, transcript = list(zip(*instances))

        empty = torch.zeros(0)

        return Batch(empty, empty, transcript, tokens, token_lengths)
