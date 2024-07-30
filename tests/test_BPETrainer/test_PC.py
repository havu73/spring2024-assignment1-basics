import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from BPETrainer.Word import Word
from BPETrainer.pair_count import PC
from collections import Counter
import unittest
import pytest

class TestPC(unittest.TestCase):

    def test_init(self):
        word_list = 'low ' * 5 + 'lower ' * 2 + 'newest ' * 6 + 'widest ' * 3
        word_list = word_list.split()
        word_count = Counter(word_list)
        Word_list = []
        word_Word = {}
        for word in word_count:
            Word_list.append(Word(word, word_count[word]))
            word_Word[word] = Word_list[-1]
        pc = PC(word_list=Word_list)
        exp_pc = Counter({('l', 'o'): 7, ('o', 'w'): 7, ('w', 'e'): 8, ('e', 'r'): 2, ('w', 'i'): 3,
                          ('i', 'd'): 3, ('d', 'e'): 3, ('e', 's'): 9, ('s', 't'): 9, ('n', 'e'): 6, ('e', 'w'): 6})
        self.assertEqual(pc.pc, exp_pc)
        max_pair = ('s', 't')
        max_count = 9
        self.assertEqual(pc.max_pair, max_pair)
        self.assertEqual(pc.max_count, max_count)

    def test_eject_max_pair(self):
        word_list = 'low ' * 5 + 'lower ' * 2 + 'newest ' * 6 + 'widest ' * 3
        word_list = word_list.split()
        word_count = Counter(word_list)
        Word_list = []
        word_Word = {}
        for word in word_count:
            Word_list.append(Word(word, word_count[word]))
            word_Word[word] = Word_list[-1]
        pc = PC(word_list=Word_list)
        max_pair, max_count = pc.eject_max_pair()
        exp_max_pair = ('s', 't')
        exp_max_count = 9
        self.assertEqual(max_pair, exp_max_pair)
        self.assertEqual(max_count, exp_max_count)
        words_with_pair = [word_Word['newest'], word_Word['widest']]
        for word in words_with_pair:
            pair_delta = word.merge_token(exp_max_pair[0], exp_max_pair[1])
            # print(word.word)
            # print(pair_delta)
            pc.update_pair_count(pair_delta, exclude_pair=max_pair)
        exp_pc = Counter({('l', 'o'): 7, ('o', 'w'): 7, ('w', 'e'): 8, ('e', 'r'): 2, ('w', 'i'): 3,
                          ('i', 'd'): 3, ('d', 'e'): 3, ('e', 'st'): 9, ('n', 'e'): 6, ('e', 'w'): 6,
                          ('e', 's'): 0})
        # print(pc.pc)
        self.assertEqual(pc.pc, exp_pc)
        self.assertEqual(pc.max_pair, ('e', 'st'))
        self.assertEqual(pc.max_count, 9)
        # another pop
        max_pair, max_count = pc.eject_max_pair()
        exp_max_pair = ('e', 'st')
        exp_max_count = 9
        self.assertEqual(max_pair, exp_max_pair)
        self.assertEqual(max_count, exp_max_count)
        words_with_pair = [word_Word['newest'], word_Word['widest']]
        for word in words_with_pair:
            pair_delta = word.merge_token(exp_max_pair[0], exp_max_pair[1])
            # print(word.word)
            # print(pair_delta)
            pc.update_pair_count(pair_delta, exclude_pair=max_pair)
        exp_pc = Counter({('l', 'o'): 7, ('o', 'w'): 7, ('w', 'e'): 2, ('e', 'r'): 2, ('w', 'i'): 3,
                          ('i', 'd'): 3, ('d', 'e'): 0, ('d', 'est'): 3, ('w', 'est'): 6, ('n', 'e'): 6, ('e', 'w'): 6,
                          ('e', 's'): 0})
        print(pc.pc)
        self.assertEqual(pc.pc, exp_pc)
        self.assertEqual(pc.max_pair, ('o', 'w'))
        self.assertEqual(pc.max_count, 7)






if __name__ == '__main__':
    unittest.main()

