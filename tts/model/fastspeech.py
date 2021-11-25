from tts.model.config import FSConfig
from .fs_layers import Encoder, Decoder, LengthAligner
from torch import nn
from tts.collate_fn import Batch


class FastSpeech(nn.Module):
    def __init__(self, config: FSConfig):
        super().__init__()

        self.encoder = Encoder(config)

        self.length_aligner = LengthAligner(config)

        self.decoder = Decoder(config)

        self.postnet = nn.Linear(config.hidden_size, config.n_mels)

    def forward(self, batch: Batch):
        x = self.encoder(batch.tokens)
       #  (batch_size, seq_len, hidden)

        x, lengths = self.length_aligner(x)

        x = self.decoder(x)

        out = self.postnet(x)

        batch.duration_prediction = lengths
        batch.melspec_prediction = out

        return batch


