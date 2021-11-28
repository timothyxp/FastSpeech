from torch.utils.data import Dataset
import torchaudio
from typing import List


default_text = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
]


class TextDataset(Dataset):
    def __init__(self, texts: List[str] = None):
        super().__init__()
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

        self.texts = texts or default_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int):
        tokens, token_lengths = self._tokenizer(self.texts[index])

        return tokens, token_lengths, self.texts[index]

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)

        return result
