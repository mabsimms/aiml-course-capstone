""" Manage Unicode friendly tokenizer for the DNN model """

import pandas as pd

from pathlib import Path
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

# Use the HuggingFace tokenizer to provide robust support for Unicode
# (TensorFlow's default tokenizer does not correctly handle Unicode, 
# expecially with embedded control characters).
# https://huggingface.co/learn/llm-course/en/chapter6/8
def train_tokenizer(
    text: pd.Series,
    vocab_size: int = 20_000,
    output_sequence_length = 3_000
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD_TOKEN, UNK_TOKEN],
        initial_alphabet=ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(text.tolist(), trainer=trainer)

    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN),
        pad_token=PAD_TOKEN,
        length=output_sequence_length
    )

    tokenizer.enable_truncation(max_length=output_sequence_length)
    return tokenizer

def load_tokenizer(path : str = "tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)

def save_tokenizer(tokenizer : Tokenizer, path : Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
