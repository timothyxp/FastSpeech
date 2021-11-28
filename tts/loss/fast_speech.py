from torch import Tensor, nn
from tts.collate_fn import Batch
from typing import Tuple


class FastSpeechLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.length_mse = nn.MSELoss()
        self.mel_spec_mse = nn.MSELoss()

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        len_mse = self.length_mse(batch.durations, batch.duration_prediction.exp())

        min_len = min(batch.melspec.shape[-1], batch.melspec_prediction.shape[-2])
        mel_spec_mse = self.mel_spec_mse(
            batch.melspec[:, :, :min_len],
            batch.melspec_prediction.transpose(-1, -2)[:, :, :min_len]
        )

        return len_mse, mel_spec_mse
