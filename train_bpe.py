import argparse
import os
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BPETrainer.bpe_trainer import BPETrainer
from Tokenizer.tokenizer import Tokenizer

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # read in the text file
    with open(input_path, "r") as f:
        text = f.read()
    # pre-tokenize the text file
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = re.findall(PAT, text)
    # pretokens = text.split()
    # train the BPE tokenizer
    bpe_train = BPETrainer(pretokens=pretokens, vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = bpe_train.train()
    bpe_train.save_trained_BPE(vocab_path=kwargs['vocab_path'], merges_path=kwargs['merges_path'])
    return vocab, merges

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "This code take in a corpus and train a BPE tokenizer.")
    parser.add_argument('--input_path', required=True, type=str, help='Path to the input corpus, such as tiny stories.')
    parser.add_argument('--vocab_size', required=True, type=int, help='The size of the vocabulary.')
    parser.add_argument('--special_tokens', required=False, type=str, nargs='+', default = ["<|endoftext|>"], help='Special tokens to be added to the vocabulary.')
    parser.add_argument('--vocab_path', required=True, type=str, help='Where to save the OUTPUT vocabulary.')
    parser.add_argument('--merges_path', required=True, type=str, help='Where to save the OUTPUT merges.')
    args = parser.parse_args()
    run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
    )
    print ('train_bpe.py: Done training BPE tokenizer.')