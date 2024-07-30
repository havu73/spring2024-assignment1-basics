from collections import Counter
from typing import List, Tuple, Dict, Iterable, Iterator
import json
import regex as re
from .word_encoder import WordEncoder

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.reverse_merges = Tokenizer._reverse_merges(merges)  # Dict[bytes, List[bytes]] --> for each byte as key of the first b, the list of bytes that it merges to
        self.reverse_vocab = Tokenizer._reverse_vocab(vocab)  # Dict[bytes, int] --> for each byte, the integer it corresponds to there is only one-to-one mapping between bytes and integers
        self.special_tokens = self._check_special_tokens(special_tokens) # this will create self.special_tokens: List[str]

    def _check_special_tokens(self, special_tokens: List[str]) -> None:
        '''
        Check if the special tokens are in the vocab
        '''
        if special_tokens is None:
            return
        # first sort the special token by decreasing length. This is so that if there are any special token nested inside another, it will come after the outer one
        special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
        for token in special_tokens:
            if token.encode('utf-8') not in self.reverse_vocab:
                raise ValueError(f'Special token {token} is not in the vocab')
        return special_tokens

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
            if len(merge) != 2:
                print(f'Merge file is not formatted correctly: {merge}')
                continue
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
    def _presplit_by_special_tokens(text: str, special_tokens: List[bytes]) -> List[str]:
        '''
        Given a text and a list of special tokens, split the text by the special tokens
        '''
        import re
        PAT = r'(?:' + r'|'.join(map(re.escape, special_tokens)) + r')'  # bc of how special_tokens are ordered, this will make sure that any special token that include another one will be matched first (such as !! will be matched before !)
        token_found = re.findall(PAT, text)
        between_tokens = re.split(PAT, text)
        return token_found, between_tokens

    def _encode_noStok_string(self, text: str) -> List[int]:
        '''
        Given a string that we know does not contain any special token, encode this string into a list of integers based on the merges and vocab
        '''
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens = re.findall(PAT, text)
        if len(pretokens) == 0:
            return []
        from collections import Counter
        pretoken_to_encode = Counter(pretokens)  # bc there may be a lot of pretokens that are exactly the same, we only need to encode each of the key in this Counter once  # Dict[str, int]
        encoded_pretokens = {}
        for pretoken in pretoken_to_encode:
            if pretoken == '':
                continue
            word_encoder = WordEncoder(pretoken, self.reverse_vocab, self.reverse_merges)
            encoded_ints = word_encoder.encode()
            encoded_pretokens[pretoken] = encoded_ints
        encoded_text = []
        for pretoken in pretokens:
            if pretoken != '':
                encoded_text.extend(encoded_pretokens[pretoken])
        return encoded_text

    def encode(self, text: str) -> List[int]:
        '''
        Given a string (most likely a word a chunk of text split by the pretokenizers,
        encode this string into a list of integers
        '''
        if self.special_tokens is None:
            return self._encode_noStok_string(text)
        Stokens_found, between_Stokens = Tokenizer._presplit_by_special_tokens(text, self.special_tokens)
        assert len(between_Stokens) == len(Stokens_found) + 1
        # between_Stokens is a list of strings that are between the special tokens, we will encode them
        # now we need to combine the special tokens and the words between special tokens
        encoded_text = []
        for stok_idx, Stoken in enumerate(Stokens_found):
            if between_Stokens[stok_idx] != '':
                encoded_text.extend(self._encode_noStok_string(between_Stokens[stok_idx]))
            encoded_text.append(self.reverse_vocab[Stoken.encode('utf-8')])
        if between_Stokens[-1] != '':
            encoded_text.extend(self._encode_noStok_string(between_Stokens[-1]))
        return encoded_text


    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        for t in text:
            encoded_list = self.encode(t)
            for encoded_int in encoded_list:
                yield encoded_int

    def decode(self, ids: List[int]) -> str:
        '''
        Given a list of integers, decode them into a string
        '''
        result = []
        replacement_byte = '\uFFFD'.encode('utf-8')
        for i in ids:
            if i in self.vocab:
                result.append(self.vocab[i])
            else:
                result.append(f'{replacement_byte}')
                print(f'Unrecognized token: {i}')
        result = b''.join(result)
        return result.decode('utf-8', errors='replace')

