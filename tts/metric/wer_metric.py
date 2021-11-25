from typing import List

import torch
from torch import Tensor
import numpy as np

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamWERMetric(BaseMetric):
    def __call__(self, beam_text: List[str], text: List[str], *args, **kwargs):
        wers = []
        for beam_text, text in zip(beam_text, text):
            wers.append(calc_wer(text, beam_text))

        return np.mean(wers)
