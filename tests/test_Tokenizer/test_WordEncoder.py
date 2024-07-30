import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Tokenizer.word_encoder import WordEncoder
from Tokenizer.tokenizer import Tokenizer
from collections import Counter
import unittest
import pytest

class TestWordEncoder(unittest.TestCase):

    def test1(self):
        vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
        reverse_vocab = Tokenizer._reverse_vocab(vocab)
        merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
        reverse_merges = Tokenizer._reverse_merges(merges)
        print(reverse_merges)
        exp_reverse_merges = {b't': [(b'h', 0)], b' ': [(b'c', 1), (b'a', 2)],
                              b'th': [(b'e', 3)],
                              b' a': [(b't', 4)]}
        self.assertEqual(reverse_merges, exp_reverse_merges)
        word = WordEncoder('the', reverse_vocab, reverse_merges)
        word_int = word.encode()
        self.assertEqual(word_int, [9])
        print(f'pass test {word.word}')
        word = WordEncoder(' cat', reverse_vocab, reverse_merges)
        word_int = word.encode()
        print(' cat', word_int)
        self.assertEqual(word_int, [7, 1, 5])
        print(f'pass test {word.word}')
        word = WordEncoder(' ate', reverse_vocab, reverse_merges)
        word_int = word.encode()
        print(f' ate', word_int)
        self.assertEqual(word_int, [10, 3])
        print(f'pass test {word.word}')
        word = WordEncoder('the cat ate', reverse_vocab, reverse_merges)
        word_int = word.encode()
        print(f'the cat ate', word_int)
        self.assertEqual(word_int, [9, 7, 1, 5, 10, 3])
        print(f'pass test {word.word}')

