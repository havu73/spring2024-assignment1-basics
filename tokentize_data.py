import argparse
import helper
import numpy as np
import json
import os
from typing import Optional
from Tokenizer.tokenizer import Tokenizer
from tests.common import gpt2_bytes_to_unicode

def load_key_value_file(file_path):
    result_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            token_int, str_val = line.strip().split()  # 10 ĉ then 10 is key, ĉ is value
            result_dict[str_val] = int(token_int)
    return result_dict
def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: Optional[list[str]] = None,
):
    '''
    Function written by Nelson Liu, course instructor, modified by me
    Given the path to the vocab file and the path to the merges file, return a tokenizer object.
    first, gpt2_byte_decoder is used such that the 256 numbers are mapped to the unicode characters. it is a little more special than just having {int --> bytes} mapping, because some bytes are not readable to the human eyes so it got converted to some other weird characters but at least more visible to the human eyes
    :param vocab_path: str | os.PathLike, path to the vocab file
    :param merges_path: str | os.PathLike, path to the merges file
    :param special_tokens: Optional[list[str]], list of special tokens
    '''
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    gpt2_vocab = load_key_value_file(vocab_path)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)

def tokenize_by_batch(input_fn, tokenizer, output_fn, buffer_size=10000):
    '''
    Tokenize the input file by batch and write the tokenized data to the output file
    :param input_fn: str, the file that we want to tokenize
    :param tokenizer: Tokenizer, the tokenizer object
    :param output_fn: str, the file that contains the tokenized data
    '''
    inF = helper.open_file(input_fn)  # this can handle both the case when the input_fn is a .gz file or a normal file
    outF = open(output_fn, "ab")
    buffer = []
    for line in inF:
        tokenized_line = tokenizer.encode(line)
        buffer.extend(tokenized_line)
        if len(buffer) >= buffer_size:
            array_to_save = np.array(buffer, dtype=np.int32)
            np.save(outF, array_to_save)
            buffer.clear()
    if buffer:  # after the for loop, if there are still some tokens in the buffer, save them
        array_to_save = np.array(buffer, dtype=np.int32)
        np.save(outF, array_to_save)
        buffer.clear()
    inF.close()
    outF.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This code will take in different possible values of parameters create design matrix dataframes that tell us the different parameter combinations that we will use to simulate the A matrix and the b vector and the solver settings.")
    parser.add_argument("--input_fn", type=str, required=True, help="the file that we want to tokenize")
    parser.add_argument("--vocab_json", type=str, required=True, help="path to the file that contains all the tokens")
    parser.add_argument("--merges_fn", type=str, required=True, help="path to the file that contains all the merges")
    parser.add_argument("--output_fn", type=str, required=True, help="path to the file that contains the tokenized data")
    parser.add_argument("--buffer_size", type=int, required=False, default=10000, help="the size of the buffer")
    args = parser.parse_args()
    helper.check_file_exist(args.input_fn)
    helper.check_file_exist(args.vocab_json)
    helper.check_file_exist(args.merges_fn)
    helper.create_folder_for_file(args.output_fn)
    # delete the output_fn if it exists
    if os.path.exists(args.output_fn):
        os.remove(args.output_fn)
    print ("Done getting command line arguments")
    tokenizer = get_tokenizer_from_vocab_merges_path(args.vocab_json, args.merges_fn)
    tokenize_by_batch(args.input_fn, tokenizer, args.output_fn, buffer_size=args.buffer_size)
    print("Done tokenizing the data")

# python tokentize_data.py --input_fn '/gladstone/engelhardt/lab/hvu//cs336/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt.gz' --vocab_json '/gladstone/engelhardt/lab/hvu/cs336/spring2024-assignment1-basics/data/vocab.json' --merges_fn '/gladstone/engelhardt/lab/hvu//cs336/spring2024-assignment1-basics/data/merges.txt' --output_fn '/gladstone/engelhardt/lab/hvu//cs336/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.npy'