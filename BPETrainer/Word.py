from collections import Counter
from typing import List, Tuple, Dict

class Word:
    def __init__(self, word: str, word_freq: int):
        self.word = word
        self.word_freq = word_freq
        self.token_list = list(word)
        self.pair_list = [(self.token_list[i], self.token_list[i+1]) for i in range(len(self.token_list)-1)]
        self.pair_freq = Counter(self.pair_list)
        self.pair_freq = Counter({pair: freq*self.word_freq for pair, freq in self.pair_freq.items()})

    def merge_token(self, token1: str, token2: str):
        '''
        Given two tokens, merge them in the token_list and pair_list
        :param token1: str
        :param token2: str
        :return: pair_delta: Counter[(str, str), int] --> dictionary of pair and how much their frequency should change given the merge
        '''
        result = self.pair_freq.copy()
        # assert token1token2 is substring of word
        assert token1+token2 in self.word, f'The tokens {token1} and {token2} are not found in the word {self.word}'
        joint_token = token1+token2
        i=0
        while i < len(self.token_list)-1: # there can be multiple occurence of the merge pattern in the word
            pair_delta = Counter({})
            if self.token_list[i] == token1 and self.token_list[i+1] == token2:  # spotted a merge place
                pair_delta.update({(token1, token2): -self.word_freq})
                self.token_list[i] = token1+token2
                self.token_list.pop(i+1)
                if i >0:
                    before_token = self.token_list[i-1]
                    add_before_pair = (before_token, self.token_list[i]) # (str, str), break when the merge pattern is at the beginning of the word
                    sub_before_pair = (before_token, token1)  # (str, str), break when the merge pattern is at the beginning of the word
                    pair_delta.update({add_before_pair: self.word_freq, sub_before_pair: -self.word_freq})
                if i < len(self.token_list)-1:  # there are still space after the merged token
                    add_after_pair = (self.token_list[i], self.token_list[i+1])  # (str, str), break when the merge pattern is at the end of the word
                    sub_after_pair = (token2, self.token_list[i+1])  # (str, str), break when the merge pattern is at the end of the word
                    pair_delta.update({add_after_pair: self.word_freq, sub_after_pair: -self.word_freq})
            # dont forget to update i and pair_freq
            self.pair_freq.update(pair_delta)
            i+=1
        self.pair_freq = Counter({pair: freq for pair, freq in self.pair_freq.items() if freq != 0})
        # update pair_list and pair_freq
        self.pair_list = [(self.token_list[i], self.token_list[i+1]) for i in range(len(self.token_list)-1)]
        result.subtract(self.pair_freq)
        result = Counter({pair: -freq for pair, freq in result.items() if freq != 0})
        return result  # Counter[(str, str), int] pair_freq_after_merge.update(pair_delta) = pair_freq before merge
