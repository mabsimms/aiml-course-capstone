import csv
from pathlib import Path
import pytest
import pandas as pd

from capstone.tokenization.tokenization import train_tokenizer, save_tokenizer, load_tokenizer, PAD_TOKEN, UNK_TOKEN

def test_tokenizer_handles_unicode():
    samples = pd.Series([
        "Hello world, this is plain ASCII text.",
        "Emoji test: \U0001F600\U0001F4A9\U0001F680",
        "Accented text: café, naïve, jalapeño",
        "CJK text: 你好，世界",
        "Control chars: line1\nline2\ttabbed",
        "Null byte: before\x00after",
        "Other C0 controls: \x01\x02\x0b\x0c\x1f",
        "Zero-width space: a\u200bb",
        "Byte order mark: \ufeffleading",
    ])

    tokenizer = train_tokenizer(samples, vocab_size=500, output_sequence_length=64)

    for test_string in [ 
         "café \U0001F600",
         "before\x00middle\x0btab\x1fend"
    ]:
        encoding = tokenizer.encode(test_string)
        decoded = tokenizer.decode(encoding.ids)

        # The pre_tokenizer is set to add a leading space to the first word, allowing treating the 
        # first word the same as any other word (https://huggingface.co/docs/tokenizers/en/api/pre-tokenizers)
        expected = test_string if test_string.startswith(" ") else " " + test_string        
        assert expected == decoded

def test_tokenizer_round_trip(tmp_path):
    samples = pd.Series([
        "Hello world, this is plain ASCII text.",
        "Emoji test: \U0001F600\U0001F4A9\U0001F680",
        "Accented text: café, naïve, jalapeño",
        "CJK text: 你好，世界",
        "Control chars: line1\nline2\ttabbed",
        "Null byte: before\x00after",
        "Other C0 controls: \x01\x02\x0b\x0c\x1f",
        "Zero-width space: a\u200bb",
        "Byte order mark: \ufeffleading",
    ])
    
    tokenizer = train_tokenizer(samples, vocab_size=500, output_sequence_length=64)

    tokenizer_path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, tokenizer_path)

    loaded_tokenizer = load_tokenizer(str(tokenizer_path))

    for test_string in [ 
            "café \U0001F600",
            "before\x00middle\x0btab\x1fend"
    ]:
        encoding = loaded_tokenizer.encode(test_string)
        decoded = loaded_tokenizer.decode(encoding.ids)

        # The pre_tokenizer is set to add a leading space to the first word, allowing treating the 
        # first word the same as any other word (https://huggingface.co/docs/tokenizers/en/api/pre-tokenizers)
        expected = test_string if test_string.startswith(" ") else " " + test_string        
        assert expected == decoded

    assert tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
