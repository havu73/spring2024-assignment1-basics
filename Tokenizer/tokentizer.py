from collections import Counter
from typing import List, Tuple, Dict, Iterable, Iterator
import json

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[int] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_merges = Tokenizer._reverse_merges(merges)  # Dict[bytes, List[bytes]] --> for each byte as key of the first b, the list of bytes that it merges to
        self.reverse_vocab = Tokenizer._reverse_vocab(vocab)  # Dict[bytes, int] --> for each byte, the integer it corresponds to there is only one-to-one mapping between bytes and integers

    @staticmethod
    def _reverse_merges(merges: List[Tuple[bytes, bytes]]) -> Dict[bytes, List[Tuple[bytes, int]]]:
        '''
        Given a list of merges, return a dictionary of the reverse merges
        :param merges: List[Tuple[bytes, bytes]] --> list of merges
        :return: Dict[bytes, List[Tuple[bytes, int]]] --> for each byte as key of the first b, the list of bytes that it merges to and the index of the merge in the merges list (lower index means the merge should happen earlier)
        '''
        reverse_merges = {}
        for merge_idx, merge in enumerate(merges):  # byte, byte
            if merge[0] in reverse_merges:
                reverse_merges[merge[0]].append((merge[1], merge_idx))
            else:
                reverse_merges[merge[0]] = [(merge[1], merge_idx)]
        return reverse_merges

    @staticmethod
    def _reverse_vocab(vocab: Dict[int, bytes]) -> Dict[bytes, int]:
        '''
        Given a vocab, return a dictionary of the reverse vocab
        '''
        reverse_vocab = {}
        for k, v in vocab.items():
            if k in reverse_vocab:
                raise ValueError('Duplicate bytes key in vocab')
            reverse_vocab[v] = k
        return reverse_vocab

    @staticmethod
    def read_merges(merges_fn: str) -> List[Tuple[bytes, bytes]]:
        merges_fn = open(merges_fn, 'r')
        merges = []
        for line in merges_fn:
            merge = line.strip().split()
            merges.append((merge[0].encode(), merge[1].encode()))
        return merges

    @staticmethod
    def read_vocab(vocab_fn: str) -> Dict[int, bytes]:
        '''
        vocab_fn is a json file with keys: int, values: str
        '''
        f = open(vocab_fn, 'r')
        data = json.load(f)
        # Convert keys from strings to integers
        data = {int(k): v.encode('utf-8') for k, v in data.items()}
        return data
    @classmethod
    def from_files(cls, vocab_fn: str, merges_fn: str, special_tokens: List[str] = None):
        vocab= cls.read_vocab(vocab_fn)
        merges= cls.read_merges(merges_fn)
        special_tokens= [token.encode('utf-8') for token in special_tokens]
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _presplit_by_special_tokens(text: str, special_tokens: List[int]) -> List[str]:
        '''
        Given a text and a list of special tokens, split the text by the special tokens
        '''
        import re
        PAT = '|'.join(map(re.escape, special_tokens))
        token_found = re.findall(PAT, text)
        between_tokens = re.split(PAT, text)
        return token_found, between_tokens

    def _encode_str_btw_special_tokens(self, text: str) -> List[int]:
        '''
        Given a string (usually a word or a few words, but no more than that) that we know does not contain any special token, encode this string into a list of integers based on the merges and vocab
        '''
        if len(text) == 0:
            return []
        if len(text) == 1:
            return [self.vocab[text[0].encode()]]  # encode to bytes and then to int
        done_merging = False
        # get the list of possible merges
        # for each character: get a list of possible merges --> combine and rank by the order the merges were added


    def encode(self, text: str) -> List[int]:
        '''
        Given a string (most likely a word a chunk of text split by the pretokenizers,
        encode this string into a list of integers
        '''
        tokens_found, between_tokens = Tokenizer._presplit_by_special_tokens(text, self.special_tokens)
        from collections import Counter
        words_to_encode = Counter(between_tokens)  # bc there may be a lot of pretokens that are exactly the same, we only need to encode each of the key in this Counter once

        encoded_text = []



    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: List[int]) -> str:
        pass


