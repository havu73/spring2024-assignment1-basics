from collections import Counter
from typing import List, Tuple, Dict

class PC:
    def __init__(self, pc: Dict[str, int] = None, word_list: List[object] = None):
        if pc is None and word_list is not None:
            self.pc = PC.get_pair_count_from_words(word_list)
        elif pc is None and word_list is None:
            self.pc = Counter({})
        else: # pc is not None
            assert isinstance(pc, Counter), f'The input pc should be a Counter object, but it is {type(pc)}'
            self.pc=pc
        self._find_max_pair()  # self.max_pair: (str, str) --> the pair with the maximum count, self.max_count: int --> the count of the max_pair

    @staticmethod
    def get_pair_count_from_words(word_list: List[object]) -> Counter[str, int]:
        '''
        Given a dictionary of words as keys and their counts as values,
        :param word_list: List[Word] --> list of word objects
        return a dictionary of pairs of characters as keys and their counts as values.
        '''
        pair_count = Counter({})
        for word in word_list:
            pair_count.update(word.pair_freq)  # pair_freq is a Counter [(str,str), int]
        return pair_count

    def update_pair_count(self, pair_delta:Dict[Tuple[str, str], int], exclude_pair: Tuple[str, str] = None):
        '''
        Given a dictionary of character_pair as keys and delta values as values,
        update self.pc with the delta values.
        In the mean time, also update the max_count and max_pair.
        '''
        for pair in pair_delta:
            if exclude_pair is not None and pair == exclude_pair:  # we can skip the pair that we want to exclude, this probably corresponds to the pair that we just popped
                continue
            if pair not in self.pc:
                self.pc[pair] = 0
            self.pc[pair] += pair_delta[pair]
        self._find_max_pair() # there is no point in trying to find the max pair while we update, there are logical holes
        return

    def _find_max_pair(self):
        '''
        Given the current pair_count, find the pair with the maximum count and its corresponding count.
        '''
        if len(self.pc) == 0:
            self.max_count = 0
            self.max_pair = None
            return None, None
        max_count=0
        max_pair=None
        for pair in self.pc:
            if self.pc[pair] > max_count:
                max_count = self.pc[pair]
                max_pair = pair
            if (max_pair is not None) and self.pc[pair] == max_count and pair > max_pair:  # if there is a tie, choose the pair with the bigger lexicographical order
                # python is capable of doing ("BA", "A") > ("A", "B") --> True
                max_pair = pair
        self.max_count = max_count
        self.max_pair = max_pair
        return max_pair, max_count


    def eject_max_pair(self):
        if self.max_pair is None and len(self.pc) > 0:
            _, _ = self._find_max_pair()  # this function also updates self.max_count and self.max_pair
        result_max_pair = self.max_pair  # this would not change result_max_pair if self.max_pair is changed
        result_max_count = self.max_count  # this would change result_max_count if self.max_count is changed
        # remove the max_pair from the pair_count
        self.pc.pop(self.max_pair)
        self.max_pair = None # reset
        self.max_count = None # reset
        return result_max_pair, result_max_count
