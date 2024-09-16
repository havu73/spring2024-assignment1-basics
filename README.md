# CS336 Spring 2024 Assignment 1: Basics
In this repository, I am trying to do the first homework for course [CS336 at Stanford Spring 2024](https://stanford-cs336.github.io/spring2024/), in which I get to learn to implement a large language model ***from scatch***. I do this course work in my free time to catch up with the field and sharpen my learning. I owe lots of thanks to the creators of the course for designing such comprehensive homework, and making it available for the general public. Even though I am not too far into the homework yet, I have had a lot of fun and the learning is invaluable. Here is what I have done so far:
- [x] Set up the environment on the computer cluster
- [x] Implemeted the BPETrainer class and passed unit test: added ```./BPETrainer``` and ```tests/test_BPETrainer/``` folders
- [x] Implemented the Tokenizer class and passed unit test: added ```./Tokenizer``` and ```tests/test_Tokenizer/``` folders
- [x] Trained BPETraner for TinyStories dataset
- [x] Transformer language model. All code for this part is in the ```./Transformer``` folder. Implemented: 
  - [x] Root mean square layer normalization
  - [x] Position-wise Feed-Forward Networks
  - [x] Softmax
  - [x] Scaled dot product attention: in a forward call, ```mask #(seq_len, seq_len)``` is implemented such that if ```mask[i, j] = 1``` then the attention score between token ```i``` and token ```j``` is zero, so in a mask matrix, $mask[i, j] = 1$ if we ignore token $j$ when do inference at token $i$.
  - [x] Multi-head self-attention: the provided test by the course instructor is *not* correct! I made my own test to compare my implementation's result with the output of ```torch.nn.MultiheadAttention```, ```pytest -k test_multihead_self_attention_ha```. Note that in both my own implementation of multi-head self-attention and in ```torch.nn.MultiheadAttention```, masking is set such that $1$ means *masking OUT*. Therefore, in LLM contexts, for each current token, we only wants the model to attend to preceding tokens and not the following tokens. Therefore, the mask matrix of size ```(seq_len, seq_len)``` should be upper triangular with diagonal entries being $0$. In my own implementation, the mask is automatically set for such a case.  This unit testing really screwed me up so much time!
  - [x] Transformer block: failed the unit test but I am pretty sure the test by the course instructor is not correct. The test basically provides input, output, weights and compared their output to my output. However,  the Multi-head self-attention's test is not correct (my implementation was correct if compared to ```torch.nn.MultiHeadAttention``` output but failed the course instructors' provided test). Therefore, it is not surprising that my implementation of the Transformer block failed the provided test. I am confident that my implementation is correct, and due to time limit, will move on.
  - [x] The full Transformer model:  Did not pass the provided test, due to the same reasons outlind right above.
  - [x] Accounting of memory and time complexity for Transformer model. Detailed accounting is logged within the code. In general, GPT-2 XL model needs ~ 107MB/layer and ~ 27 Billions FLOPs/layer. If context length beccomes 16,384 then the model needs ~ 57 billions FLOPs/layer. All these numbers are for ONE sample (of length ```seq_len```) and ONE layer.
- [x] Optimizer Implementations:
  - [x] Cross-entropy loss
  - [x] Stochastic Gradient Descent Optimizer: ```__init__```, ```step``` functions
  - [x] AdamW Optimizer: ```__init__```, ```step``` functions: passed!
  - [x] Accounting of memory and time complexity for AdamW:
    - It takes ~12.8GB of memory to store the states of the AdamW optimizer for GPT-2 XL model. This number does not take into account the space needed to calculate the loss function and gradients (though I took into acount the space to store the gradient itself, not the space to calculate the gradient).
    - Space per batch of size 1 to calculate the loss function: ~ 0.19 GB
    - Space per batch of size 1 to calculate the soft-maxed output probabilities: ~ 0.19 GB
    - Space per batch of size 1 to store the all the components of one forward pass: ~ 0.69 GB (It may be smaller, but I am assuming I have to memorize the intermediate results of each step along the forward pass)
    - If I have 80GB of memory, I can train with batch size of: ~76 (I maybe wrong, got a bit confused by the question!). But, if ```loss.backward()``` takes twice as much memory as ```optimizer.step()```, then I can probably train with batch size of 47
    - FLOPS to run one AdamW step: ~ 14 Billions
    - FLOPS to run one forward pass: ~ 3.2E12 
    - FLOPS to run one backward pass: ~6.4E12
    - FLOPs to train all the model with given batch size and num epochs: 3.9E21 FLOPs
    - If I am still correct, it takes 488 days to train the model given a single A100 GPU.
  - [x] Cosine learning rate scheduler with warm-up
  - [x] Gradient clipping
- [x] Training loop. Implement
  - [x] Dataloader (not the dataloader object, but the function to load the data into tuple format with fixed sequence length)
  - [x] Model checkpoint: ```save_checkpoint```, ```load_checkpoint``` functions
  - [x] Encode data for training loop: ```tokenize_data.py```. My tokenizer is extremely slow. It is correct, but unacceptably slow.
  - [x] Training loop
- [ ] Run tokenizer on TinyStory
- [ ] Run tokenizer on OpenWebText
- [ ] Generate new texts from trained model
- [ ] Experiments
  - [ ] Experiment logging
  - [ ] Params tuning for TinyStories dataset: learning rate, batch size, generate texts
  - [ ] Implement parallel layers
  - [ ] Layer norm ablation
  - [ ] Run the model on OpenWebText dataset

For a full description of the assignment, see the assignment handout at [cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf).


## Setup
Below are the details about setting up the environment from the course instructors.
0. Set up a conda environment and install packages:

``` sh
conda create -n cs336_basics python=3.10 --yes
conda activate cs336_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

