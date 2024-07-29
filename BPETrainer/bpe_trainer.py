from collections import Counter
from typing import List, Tuple, Dict

class Word:
    def __init__(self, word: str, word_freq: int):
        self.word = word
        self.word_freq = word_freq
        self.token_list = list(word)
        self.pair_list = [(self.token_list[i], self.token_list[i+1]) for i in range(len(self.token_list)-1)]
        self.pair_freq = Counter(self.pair_list)

    def merge_token(self, token1: str, token2: str):
        '''
        Given two tokens, merge them in the token_list and pair_list
        :param token1: str
        :param token2: str
        :return: pair_delta: Counter[(str, str), int] --> dictionary of pair and how much their frequency should change given the merge
        '''
        pair_delta = Counter({})  # counter object can handle when key not in counter, can take negative counts, can take None as key
        # assert token1token2 is substring of word
        assert token1+token2 in self.word, f'The tokens {token1} and {token2} are not found in the word {self.word}'
        for i in range(len(self.token_list)-1):  # there can be multiple occurence of the merge pattern in the word
            if self.token_list[i] == token1 and self.token_list[i+1] == token2:  # spotted a merge place
                self.token_list[i] = token1+token2
                self.token_list.pop(i+1)
                try:
                    add_before_pair = (self.token_list[i-1], self.token_list[i]) # (str, str), break when the merge pattern is at the beginning of the word
                    sub_before_pair = (self.token_list[i-1], token1)  # (str, str), break when the merge pattern is at the beginning of the word
                except:
                    add_before_pair = None
                    sub_before_pair = None
                try:
                    add_after_pair = (self.token_list[i], self.token_list[i+1])  # (str, str), break when the merge pattern is at the end of the word
                    sub_after_pair = (token2, self.token_list[i+1])  # (str, str), break when the merge pattern is at the end of the word
                except:
                    add_after_pair = None
                    sub_after_pair = None
                update_dict = {add_before_pair: self.word_freq, add_after_pair: self.word_freq, sub_before_pair: -self.word_freq, sub_after_pair: -self.word_freq}
                pair_delta.update(update_dict)
                self.pair_freq.update(update_dict)
        # update pair_list and pair_freq
        self.pair_list = [(self.token_list[i], self.token_list[i+1]) for i in range(len(self.token_list)-1)]
        return pair_delta

class PC:
    def __init__(self, pc: Dict[str, int] = None, word_list: List[Word] = None):
        if pc is None and word_list is not None:
            self.pc = self.get_pair_count_from_words(word_list)
        if pc is None and word_list is None:
            self.pc = Counter({})
        else: # pc is not None
            assert isinstance(pc, Counter), f'The input pc should be a Counter object, but it is {type(pc)}'
            self.pc=pc
        self._find_max_pair()  # self.max_pair: (str, str) --> the pair with the maximum count, self.max_count: int --> the count of the max_pair

    @staticmethod
    def get_pair_count_from_words(self, word_list: List[Word]) -> Counter[str, int]:
        '''
        Given a dictionary of words as keys and their counts as values,
        :param word_list: List[Word] --> list of word objects
        return a dictionary of pairs of characters as keys and their counts as values.
        '''
        pair_count = Counter({})
        for word in word_list:
            pair_count.update(word.pair_freq)  # pair_freq is a Counter [(str,str), int]
        return pair_count

    def update_pair_count(self, pair_delta: Dict[(str, str), int]):
        '''
        Given a dictionary of character_pair as keys and delta values as values,
        update self.pair_count with the delta values.
        In the mean time, also update the max_count and max_pair.
        '''
        for pair in pair_delta:
            if pair not in self.pc:
                self.pc[pair] = 0
            self.pc[pair] += pair_delta[pair]
            if self.pc[pair] > self.max_count:
                self.max_count = self.pc[pair]
                self.max_pair = pair
            elif self.pc[pair] == self.max_count and pair > self.max_pair: # if there is a tie, we get the one with the bigger lexicographical order
                self.max_pair = pair
        return

    def _find_max_pair(self):
        '''
        Given the current pair_count, find the pair with the maximum count and its corresponding count.
        '''
        if len(self.pair_count) == 0:
            self.max_count = 0
            self.max_pair = None
            return None, None
        max_count=0
        max_pair=None
        for pair in self.pc:
            if self.pc[pair] > max_count:
                max_count = self.pc[pair]
                max_pair = pair
            if self.pc[pair] == max_count and pair > max_pair:  # if there is a tie, choose the pair with the bigger lexicographical order
                # python is capable of doing ("BA", "A") > ("A", "B") --> True
                max_pair = pair
        self.max_count = max_count
        self.max_pair = max_pair
        return max_pair, max_count


    def eject_max_pair(self) -> Tuple[(str, str), int]:
        if self.max_pair is None and len(self.pair_count) > 0:
            _, _ = self._find_max_pair()  # this function also updates self.max_count and self.max_pair
        result_max_pair = self.max_pair  # this would not change result_max_pair if self.max_pair is changed
        result_max_count = self.max_count  # this would change result_max_count if self.max_count is changed
        # remove the max_pair from the pair_count
        self.pc.pop(self.max_pair)
        # after I pop, I have to update the max_pair and max_count
        self._find_max_pair() # HAHAHA: have not found a better to do this yet, in theory I implement some sort of heapq, but I am not sure I can do it now
        return result_max_pair, result_max_count






class BPETrainer:
    def __init__(self, pretokens: List[str], vocab_size: int, ngram: int):
        # word_count: Dict[str, int] --> count of pretokens in the documents
        # Word_list: List[Word] --> list of word objects
        # word_Word: Dict[str, Word] --> word and its corresponding word object
        # pair_count: Dict[(str,str), int] --> frequency of each pair of characters
        # pair_word: Dict[(str,str), List[str]] --> each pair of characters and the list of words that this pair is found in
        # vocab: List[str] --> list of unique characters in the document
        # max_vocab_size: int --> the maximum size of the vocab

        # init in the following order:
        # word_count
        # Word_list and word_Word simultaneously
        # pair_count from Word_list
        # pair_word by simple iteration through all the words (not list of Word objects)
        self.word_count = Counter(pretokens)
        self.Word_list = []
        self.word_Word = {}
        for word in self.word_count:
            self.Word_list.append(Word(word, self.word_count[word]))
            self.word_Word[word] = self.Word_list[-1]
        self.pair_count = PC(word_list=self.Word_list)
        self.pair_word = self.initialize_pair_word(pretokens)
        self.vocab = self.get_init_vocab()
        self.max_vocab_size = vocab_size
        # train in the following order:
        # get the most frequent pair of characters (str, str) --> add to vocab the joint string
        # get the list of words that this pair is found in, from pair_word
        # for each of the corresponding Word.merge_token --> pair_delta: Dict[(str, str), int] --> dictionary of pair and how much their frequency should change within this word
        # update the pair_count with Word.pair_delta
        # update pair_word with the merged pair, such that for each word in the list, we update the pair_word with the new pair
        # repeat until we reach the max_vocab_size

    def train(self):
        '''
        Train the BPE model to get the vocab of size self.vocab_size
        '''
        while len(self.vocab) < self.max_vocab_size and len(self.pair_count) > 0:
            try:
                max_pair, max_count = self.pair_count.eject_max_pair()
                if max_count == 0 or max_pair is None:
                    break
            except:  # if we literally cannot find any pair of characters
                break
            words_with_pair = self.pair_word[max_pair]
            for word in words_with_pair:
                pair_delta = self.word_Word[word].merge_token(max_pair[0], max_pair[1])
                self.pair_count.update_pair_count(pair_delta)
                self.update_pair_word(pair_delta, word)  # update self.pair_word such that the pair that is associated with new merged pairs are added to the dictionary

    def update_pair_word(self, pair_delta:Counter[(str, str), int], word: str)-> None:
        '''
        '''
        for pair in pair_delta:
            if pair_delta[pair] > 0:  # new pair added involving the merged pair
                if pair not in self.pair_word:
                    self.pair_word[pair] = [word]
                else:
                    assert word not in self.pair_word[pair], f'The word {word} is already in the list of words for the pair {pair}. This should not happen.'
                    self.pair_word[pair].append(word)
        return

    def get_pair_delta(self, merged_pair: (str, str), pair_word: Dict[str, List[str]], word_freq: Dict[str, int]) -> Dict[str, int]:
        '''
        Given max_pair: (str,str), update the pair_delta: Dict[(str,str), int] --> dictionary of pair and how much their frequency should change
        This should be based on the words that max_pair is found in, and the frequency of these words.
        '''
        raise NotImplementedError

    def count_pretokens(self, pretokens: List[str] = None):
        '''
        Given self.pretokens (words), generate self.word_count: Dict[str, int] --> count of pretokens in the documents
        '''
        from collections import Counter
        self.word_count = Counter(pretokens)
        return


    def get_init_vocab(self) -> List[str]:
        # return the 256 by characters in the ASCII table
        return [chr(i) for i in range(256)]

    def initialize_pair_word(word_list: List[str]):
        """
        given self.pair_count and self.word_freq, generate the pair_word: Dict[str, List[str]]
        --> each pair of characters and the list of words that this pair is found in
        """
        pair_word = {}  # Dict[(str,str), List[str]] --> each pair of characters and the list of words that this pair is found in

        for word in word_list:
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                if pair not in pair_word:
                    pair_word[pair] = []
                pair_word[pair].append(word)
        self.pair_word = pair_word
        return self.pair_word