
import re
import string
from typing import List
import torch
from collections import Counter
from pyvi import ViTokenizer
from omegaconf import OmegaConf 
from constant import EMOJI_SENTIMENT_MAP
from tqdm.auto import tqdm
from joblib import Parallel, delayed

class TextPreprocessor:
    """
    A class for preprocessing text data, including cleaning, tokenization, and vocabulary building.

    Attributes:
        config (Config): Configuration object containing settings like VOCAB_SIZE and MAX_LENGTH.
        tokenizer (function): Tokenizer function (default is ViTokenizer.tokenize).
        vocab (Optional[dict]): Vocabulary dictionary mapping tokens to indices.
    """

    def __init__(self, config: OmegaConf):
        """
        Initializes the TextPreprocessor with the given configuration.

        Args:
            config (Config): Configuration object containing preprocessing settings.
        """
        self.config = config
        self.tokenizer = ViTokenizer.tokenize
        self.vocab = None

    def preprocess_text(self, input_text: str) -> str:
        """
        Preprocesses the input text by performing the following steps:
        1. Removes URLs.
        2. Removes HTML tags.
        3. Removes punctuation and digits.
        4. Replaces emojis with their corresponding sentiment words.
        5. Removes remaining emojis.
        6. Normalizes whitespace.
        7. Converts text to lowercase.

        Args:
            input_text (str): The raw input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'https?://\S+|www\.\S+', '', input_text)

        text = re.sub(r'<.*?>', '', text)

        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

        for emoji, sentiment in EMOJI_SENTIMENT_MAP.items():
            text = text.replace(emoji, sentiment)

        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r" ", text)

        text = " ".join(text.split())
        text = text.lower()

        return text

    def build_vocab(self, texts: List[str]) -> None:
        """
        Builds a vocabulary from a list of texts in parallel.

        Args:
            texts (List[str]): List of raw texts to build the vocabulary from.
        """
        def process_text(text):
            preprocessed_text = self.preprocess_text(text)
            tokens = self.tokenizer(preprocessed_text).split()
            return tokens

        token_lists = Parallel(n_jobs=4, backend='loky')(delayed(process_text)(text) for text in tqdm(texts))

        token_counter = Counter([token for tokens in token_lists for token in tokens])

        vocab = {"<pad>": 0, "<unk>": 1, "<cls>": 2}

        for token, freq in token_counter.items():
            if freq >= 2 and len(vocab) < self.config.model.vocab_size:
                vocab[token] = len(vocab)

        self.vocab = vocab

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encodes the input text into a tensor of token indices.

        Args:
            text (str): The raw input text to encode.

        Returns:
            torch.Tensor: A tensor of token indices with shape (MAX_LENGTH,).

        Raises:
            ValueError: If the vocabulary has not been built.
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")

        preprocessed_text = self.preprocess_text(text)
        tokens = self.tokenizer(preprocessed_text).split()

        encoded = [self.vocab["<cls>"]] + [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        if len(encoded) < self.config.model.max_length:
            encoded.extend([self.vocab["<pad>"]] * (self.config.model.max_length - len(encoded)))
        else:
            encoded = encoded[:self.config.model.max_length]

        return torch.tensor(encoded, dtype=torch.long)