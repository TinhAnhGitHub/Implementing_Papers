from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from data_func import TextPreprocessor



class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        preprocessor: TextPreprocessor
    ):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.preprocessor.build_vocab(texts)
    
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_text = self.preprocessor.encode_text(text)
        actual_length = (encoded_text != self.preprocessor.vocab["<pad>"]).sum().item()
        return encoded_text, actual_length, label
    


def collate_fn(batch):
    encoded_texts, lengths, labels = zip(*batch)
    padded_texts = torch.stack(encoded_texts)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return padded_texts, lengths, labels