import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from BPETrainer.Word import Word
from BPETrainer.pair_count import PC
from BPETrainer.bpe_trainer import BPETrainer
from collections import Counter
import unittest
import pytest

class TestBPE(unittest.TestCase):

    def test_merges(self):
        fn = 'example_text.txt'
        with open(fn, 'r') as f:
            text = f.read()
        pretokens = text.split()  # bc unlike the instructions, we get Sennrich et al. 2016 pretokenization separated by white space
        bpe_train = BPETrainer(pretokens=pretokens, vocab_size=257+6, special_tokens=["<|endoftext|>"])
        vocab, merges = bpe_train.train()
        ref_merges = [('s', 't'), ('e', 'st'), ('o', 'w'),
                      ('l', 'ow'), ('w', 'est'), ('n', 'e')]
        ref_merges = BPETrainer._convert_merges_to_bytes(ref_merges)
        print(merges)
        print(ref_merges)
        assert merges == ref_merges







if __name__ == '__main__':
    unittest.main()

