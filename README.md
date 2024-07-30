# CS336 Spring 2024 Assignment 1: Basics
In this repository, I am trying to do the first homework for course [CS336 at Stanford Spring 2024](https://stanford-cs336.github.io/spring2024/), in which I get to learn to implement a large language model ***from scatch***. I do this course work in my free time to catch up with the field and sharpen my learning. I owe lots of thanks to the creators of the course for designing such comprehensive homework, and making it available for the general public. Even though I am not too far into the homework yet, I have had a lot of fun and the learning is invaluable. Here is what I have done so far:
- [x] Set up the environment on the computer cluster
- [x] Implemeted the BPETrainer class and passed unit test: added ```./BPETrainer``` and ```tests/test_BPETrainer/``` folders
- [x] Implemented the Tokenizer class and passed unit test: added ```./Tokenizer``` and ```tests/test_Tokenizer/``` folders
- [ ] Transformer language model. Implement
  - [ ] Root mean square layer normalization
  - [ ] Position-wise Feed-Forward Networks
  - [ ] Softmax
  - [ ] Scaled dot product attention
  - [ ] Multi-head self-attention
  - [ ] Transformer block
  - [ ] The full Transformer model
  - [ ] Accounting of memory and time complexity for Transformer model
- [ ] Optimizer Implementations:
  - [ ] Cross-entropy loss
  - [ ] Stochastic Gradient Descent Optimizer: ```__init__```, ```step``` functions
  - [ ] AdamW Optimizer: ```__init__```, ```step``` functions
  - [ ] Accounting of memory and time complexity for AdamW
  - [ ] Cosine learning rate scheduler with warm-up
  - [ ] Gradient clipping
- [ ] Training loop. Implement
  - [ ] Training loop
  - [ ] Model checkpoint: ```save_checkpoint```, ```load_checkpoint``` functions
  - [ ] Run the training loop
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

