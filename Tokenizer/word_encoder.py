from collections import Counter
from typing import List, Tuple, Dict, Iterable, Iterator
import json
import regex as re

class WordEncoder:
    def __init__(self, word: str, reverse_vocab: Dict[bytes, int], reverse_merges: Dict[bytes, List[Tuple[bytes, int]]]):
        self.word = word
        self.byte_list = [bytes([c]) for c in word.encode()]
        self.reverse_merges = reverse_merges  # Dict[bytes, List[Tuple[bytes, int]]] --> for each byte as key of the first b, the list of bytes that it merges to
        self.reverse_vocab = reverse_vocab  # Dict[bytes, int] --> for each byte, the integer it corresponds to there is only one-to-one mapping between bytes and integers
        self.possible_merges, self.candidate_merge, self.min_merge_idx = self.get_possble_merges()
        # self.possible_merge: List[List[Tuple[bytes, int]]] --> for each byte in self.byte_list[:-1], the list of (bytes, merge_idx) that it can merge to
        self.merged_byte_list = [] # List[bytes] --> the list of bytes that we have merged so far, init to empty. This list will be filled by order of disocvery of merges

    def get_possble_merges(self)-> Tuple[List[List[Tuple[bytes, int]]], Tuple[bytes, bytes], int]:
        '''
        Given the current byte_list, reverse_vocab (non-changing) and reverse_merges (non-changing), return a list of possible merges for each byte and candidate merge
        '''
        result = [[]] * len(self.byte_list)  # for each byte, the list of (bytes, merge_idx) that it can merge to
        min_merge_idx = None
        candidate_merge = None
        for b_idx, b in enumerate(self.byte_list[:-1]):  # because of how the tokenrizer is trained, we would never have to worry about a case where a,nd happens before n,d
            # therefore, this WordEncoder logic of just checking the byte immediately after the current byte is correct, we do not take into account any other bytes following the next byte
            next_byte = self.byte_list[b_idx + 1]
            if b in self.reverse_merges:
                for (poss_nextB, merge_idx) in self.reverse_merges[b]:
                    if poss_nextB == next_byte:  # we have found a possible merge
                        result[b_idx].append((poss_nextB, merge_idx))
                        if min_merge_idx is None or merge_idx < min_merge_idx:  # it will never happen that merge_idx == min_merge_idx
                            min_merge_idx = merge_idx
                            candidate_merge = (b, poss_nextB)
        return result, candidate_merge, min_merge_idx

    def merge(self, candidate_merge: Tuple[bytes, bytes]):
        '''
        Merge the byte at merge_idx with the next byte
        Change self.byte_list, self.merged_byte_list
        '''
        # note that if we use regex to split the words, there maybe case that the components are ''--> ignore
        '''
        import re
        text = "Hello, world! This is an example; let's split it by the rules!!thisthe."
        # Define the dividers, ensuring more specific ones come first (later merges should come first, this is very important)
        dividers = r'(?:!!|the|th|!)'
        result = re.split(dividers, text)
        print(result)
        >>>['Hello, world', " This is an example; let's split it by ", ' rules', '', 'is', '.']
        '''
        self.merged_byte_list.append(candidate_merge[0] + candidate_merge[1])  # merge the bytes
        PAT = r'(?:' + '|'.join([re.escape(b.decode()) for b in self.merged_byte_list[::-1]]) + ')'  # this is needed because want to divide the word based on dividers of the merged bytes such that newest merged bytes are presented first,
        # newer merges include the older merges as substring.
        byte_list = re.split(PAT, self.word)
        self.byte_list = [bytes([c]) for c in byte_list if c != '']  # due to the split, there can be empty strings, we ignore them
        return

    def _convert_byte_to_int(self)-> List[int]:
        '''
        Convert the self.byte_list to a list of integers. Basically, we consider the byte_list as a list of character in a word that we will encode into integers
        '''
        result = []
        for b in self.byte_list:
            if b not in self.reverse_vocab:
                raise ValueError('Byte not in reverse_vocab')
            result.append(self.reverse_vocab[b])
        return result

    def encode(self)-> List[int]:
        '''
        Encode the word using the merges
        '''
        while self.candidate_merge is not None:
            self.merge(self.min_merge_idx, self.candidate_merge)  # update self.byte_list and self.merged_byte_list
            self.possible_merges, self.candidate_merge, self.min_merge_idx = self.get_possble_merges()
        # after we are done merging the bytes in the word, we will encode it
        return self._convert_byte_to_int()

