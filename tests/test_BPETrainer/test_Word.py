import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from BPETrainer.Word import Word
from collections import Counter
import unittest
import pytest

class TestWord(unittest.TestCase):

    def test1(self):
        word = Word('abac', 1) # {(a,b):1, (b,a):1, (a,c):1}
        sub_freq = word.merge_token('a', 'b')
        exp_pair_freq = Counter({('ab', 'a'): 1, ('a', 'c'):1})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_pair_list = [('ab', 'a'), ('a', 'c')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_token_list = ['ab', 'a', 'c']
        self.assertEqual(word.token_list, exp_token_list)
        exp_sub_freq = Counter({('a', 'b'): -1, ('b', 'a'):-1, ('ab', 'a'):1})
        self.assertEqual(sub_freq, exp_sub_freq)

    def test2(self):
        word= Word('ababc', 1)  # {(a,b):2, (b,a):1, (b,c):1}
        sub_freq = word.merge_token('a', 'b')
        exp_pair_freq = Counter({('ab', 'ab'): 1, ('ab', 'c'):1})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_pair_list = [('ab', 'ab'), ('ab', 'c')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_token_list = ['ab', 'ab', 'c']
        self.assertEqual(word.token_list, exp_token_list)
        exp_sub_freq = Counter({('a', 'b'): -2, ('b', 'a'):-1, ('ab', 'ab'):1,('ab', 'c'):1, ('b', 'c'): -1})
        self.assertEqual(sub_freq, exp_sub_freq)

    def test3(self):
        word = Word('ababab', 2)  # {(a,b):6, (b,a):4}
        sub_freq = word.merge_token('a', 'b')
        # print(word.pair_freq)
        # print(word.pair_list)
        # print(word.token_list)
        exp_pair_freq = Counter({('ab', 'ab'): 4})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_pair_list = [('ab', 'ab'), ('ab', 'ab')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_token_list = ['ab', 'ab', 'ab']
        self.assertEqual(word.token_list, exp_token_list)
        exp_sub_freq = Counter({('a', 'b'): -6, ('b', 'a'):-4, ('ab', 'ab'):4})
        self.assertEqual(sub_freq, exp_sub_freq)

    def test4(self):
        word = Word('banana', 2)  # {(b,a):2, (a,n):4, (n,a):4}
        sub_freq = word.merge_token('n', 'a')
        # print(word.pair_freq)
        # print(word.pair_list)
        # print(word.token_list)
        exp_pair_freq = Counter({('b', 'a'): 2, ('a', 'na'): 2, ('na', 'na'): 2})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_pair_list = [('b', 'a'), ('a', 'na'), ('na', 'na')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_token_list = ['b', 'a', 'na', 'na']
        self.assertEqual(word.token_list, exp_token_list)
        exp_sub_freq = Counter({('n', 'a'): -4, ('a', 'n'): -4, ('a', 'na'): 2, ('na', 'na'): 2})
        self.assertEqual(sub_freq, exp_sub_freq)

    def test5(self):
        word = Word('newest', 1) # {(n,e):1, (e,w):1, (w,e):1, (e,s):1, (s,t):1}
        sub_freq = word.merge_token('s', 't')
        exp_token_list = ['n', 'e','w', 'e', 'st']
        self.assertEqual(word.token_list, exp_token_list)
        exp_pair_list = [('n', 'e'), ('e', 'w'), ('w', 'e'), ('e', 'st')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_pair_freq = Counter({('n', 'e'): 1, ('e', 'w'): 1, ('w', 'e'): 1, ('e', 'st'): 1})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_sub_freq = Counter({('s', 't'): -1, ('e', 's'): -1, ('e', 'st'): 1})
        self.assertEqual(sub_freq, exp_sub_freq)
        sub_freq = word.merge_token('e', 'st')  # {(n,e):1, (e,w):1, (w,e):1, (e,st):1}
        exp_token_list = ['n', 'e','w', 'est']
        self.assertEqual(word.token_list, exp_token_list)
        exp_pair_list = [('n', 'e'), ('e', 'w'), ('w', 'est')]
        self.assertEqual(word.pair_list, exp_pair_list)
        exp_pair_freq = Counter({('n', 'e'): 1, ('e', 'w'): 1, ('w', 'est'): 1})
        self.assertEqual(word.pair_freq, exp_pair_freq)
        exp_sub_freq = Counter({('e', 'st'): -1, ('w', 'e'): -1, ('w', 'est'): 1})
        self.assertEqual(sub_freq, exp_sub_freq)

if __name__ == '__main__':
    unittest.main()