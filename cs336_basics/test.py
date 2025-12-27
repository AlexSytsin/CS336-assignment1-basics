from transformer import SGD
import torch
from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files("/Users/alex/CS336/CS336-assignment1-basics/tokenizers/owt_train.pkl", special_tokens=["<|endoftext|>", "<||>"])
print(len(tokenizer.vocab))
print(len(tokenizer.merges))
print(tokenizer.special_tokens)