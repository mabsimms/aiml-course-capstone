#!/usr/bin/env python3
"""
Compare whole-word length (the unit the old output_sequence_length=3000 was
tuned against) with BPE subword length (the unit the current HF tokenizer
produces), to re-derive output_sequence_length for the DNN instead of
carrying over the old constant across a change in tokenization granularity.
"""
import argparse
import numpy as np

from pathlib import Path

from capstone.cli import load_raw_data
from capstone.dataset import prepare_experiment
from capstone.tokenization.tokenization import train_tokenizer

PERCENTILES = [50, 90, 95, 99, 100]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True,
                        help="Directory of raw per-source CSVs (e.g. the kagglehub dataset dir)")
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--candidates", type=int, nargs="+",
                        default=[3000, 3500, 4000, 4500, 5000, 6000],
                        help="Candidate output_sequence_length values to evaluate")
    args = parser.parse_args()

    raw = load_raw_data(None, args.directory)
    df_train, _, _ = prepare_experiment(raw, stratify_by_source=True)
    text = df_train["text"]

    # Old unit: whitespace word count -- matches filter_low_quality_text's
    # definition in capstone/dataset.py, and what Keras TextVectorization counted.
    word_lengths = text.str.split().str.len().to_numpy()

    # New unit: BPE subword token count. Train without truncation/padding so we
    # measure each document's true length, not one already clipped to 3000.
    tokenizer = train_tokenizer(text, vocab_size=args.vocab_size)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    bpe_lengths = np.array([len(e.ids) for e in tokenizer.encode_batch(text.tolist())])

    print(f"{len(text)} documents\n")
    print(f"{'percentile':>10} {'words':>10} {'bpe tokens':>12} {'tokens/word':>12}")
    for p in PERCENTILES:
        w = np.percentile(word_lengths, p)
        b = np.percentile(bpe_lengths, p)
        print(f"{p:>10} {w:>10.0f} {b:>12.0f} {(b / w if w else float('nan')):>12.2f}")

    print(f"\n{'candidate len':>14} {'% docs fully covered':>22} {'% docs truncated':>18}")
    for length in sorted(args.candidates):
        covered = (bpe_lengths <= length).mean() * 100
        print(f"{length:>14} {covered:>21.2f}% {100 - covered:>17.2f}%")

if __name__ == "__main__":
    main()