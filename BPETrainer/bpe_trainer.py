from collections import Counter
from typing import List, Tuple, Dict
from .Word import Word
from .pair_count import PC
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
from gpt2_bytes import common


class BPETrainer:
    def __init__(self, pretokens: List[str], vocab_size: int, special_tokens: List[str] = []):
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
            if len(word)>1:  # ignore puctuation and single characters
                self.Word_list.append(Word(word, self.word_count[word]))
                self.word_Word[word] = self.Word_list[-1]
        self.pair_count = PC(word_list=self.Word_list)
        self.pair_word = self.initialize_pair_word()
        self.vocab = []
        self.max_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges = []  # list(Tuple[bytes, bytes]) --> list of merged pairs of characters
        self.trained= False
        # train in the following order:
        # get the most frequent pair of characters (str, str) --> add to vocab the joint string
        # get the list of words that this pair is found in, from pair_word
        # for each of the corresponding Word.merge_token --> pair_delta:Dict[Tuple[str, str], int] --> dictionary of pair and how much their frequency should change within this word
        # update the pair_count with Word.pair_delta
        # update pair_word with the merged pair, such that for each word in the list, we update the pair_word with the new pair
        # repeat until we reach the max_vocab_size

    def train(self):
        '''
        Train the BPE model to get the vocab of size self.vocab_size
        '''
        if self.trained:
            return BPETrainer._convert_vocab_to_bytes(vocab = self.vocab , special_tokens=self.special_tokens), BPETrainer._convert_merges_to_bytes(self.merges)
        # import pdb; pdb.set_trace()
        existing_vocab_len= len(self.special_tokens)+ 256
        while existing_vocab_len < self.max_vocab_size:
            try:
                max_pair, max_count = self.pair_count.eject_max_pair()
                if max_count == 0 or max_pair is None:
                    break
                # print(max_pair)
                # import pdb; pdb.set_trace()
                self.merges.append(max_pair)
                self.vocab.append(''.join(max_pair))
                existing_vocab_len += 1
            except:  # if we literally cannot find any pair of characters
                break
            words_with_pair = self.pair_word[max_pair]
            for word in words_with_pair:
                pair_delta = self.word_Word[word].merge_token(max_pair[0], max_pair[1])
                self.pair_count.update_pair_count(pair_delta, exclude_pair=max_pair)
                self.update_pair_word(pair_delta, word)  # update self.pair_word such that the pair that is associated with new merged pairs are added to the dictionary
        # to return, we need to return the merges and the vocab
        self.trained = True
        # merges: List[Tuple[bytes, bytes]] --> list of merged pairs of characters
        # vocab: Dict[int, bytes] --> dictionary of character index and unique characters in the document
        return BPETrainer._convert_vocab_to_bytes(vocab = self.vocab , special_tokens=self.special_tokens), BPETrainer._convert_merges_to_bytes(self.merges)

    def save_trained_BPE(self, vocab_path: str, merges_path: str) -> None:
        '''
        Save the trained BPE model to the specified paths
        '''
        vocab, merges = self.train()  # we do not need it to return the bytes, just need to form self.merges and self.vocab
        # vocab is a dictionary: key: int[0. vocab_size], value: bytes of the character(s) that the integer corresponds to
        # merges [Tuple(bytes, bytes)] --> list of merged pairs of characters
        byte_to_char = common.gpt2_bytes_to_unicode()  # we have the correct mapping int --> bytes, but in order to print the mappings out into readable file,
        # we use the gpt2's ways of mapping the 256 bytes into readable characters
        # write vocab into a json file, each line show the character, that the value is the integer corresponding to the character
        with open(vocab_path, 'w') as f:
            for ind, byte in vocab.items():
                value_to_print = ''.join([byte_to_char[b] for b in byte])
                f.write(f"{ind} {value_to_print}\n")
        f.close()
        # write merges into a text file, each line is a pair of characters
        with open(merges_path, 'w') as f:
            for merge in merges:
                merge0 = ''.join([byte_to_char[b] for b in merge[0]])
                merge1 = ''.join([byte_to_char[b] for b in merge[1]])
                f.write(merge0+' '+merge1+'\n')
        f.close()
        return


    @staticmethod
    def _convert_vocab_to_str(vocab: List[str], special_tokens:List[str]=[]) -> Dict[int, str]:
        results = special_tokens + [chr(i) for i in range(256)] + vocab
        results = {i: results[i] for i in range(len(results))}
        return results

    @staticmethod
    def _convert_vocab_to_bytes(vocab: List[str], special_tokens:List[str]=[]) -> Dict[int, bytes]:
        results = [x.encode('utf-8') for x in special_tokens]
        results.extend([x.to_bytes(1,"big") for x in range(256)])
        results.extend([x.encode('utf-8') for x in vocab])
        results = {i: results[i] for i in range(len(results))}
        return results

    @staticmethod
    def _convert_merges_to_bytes( merges: List[Tuple[str, str]]) -> List[Tuple[bytes, bytes]]:
        return [(merge[0].encode('utf-8'), merge[1].encode('utf-8')) for merge in merges]

    def update_pair_word(self, pair_delta:Counter[Tuple[str, str], int], word: str)-> None:
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



    def get_init_vocab(self, special_tokens:List[str]=[]) -> List[str]:
        # return the 256 by characters in the ASCII table
        return special_tokens + [i.to_bytes(1,"big") for i in range(256)]


    def initialize_pair_word(self):
        """
        given self.pair_count and self.word_freq, generate the pair_word: Dict[str, List[str]]
        --> each pair of characters and the list of words that this pair is found in
        """
        pair_word = {}  # Dict[(str,str), List[str]] --> each pair of characters and the list of words that this pair is found in
        for word in self.word_count.keys():
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                if pair not in pair_word:
                    pair_word[pair] = []
                pair_word[pair].append(word)
        self.pair_word = pair_word
        return self.pair_word