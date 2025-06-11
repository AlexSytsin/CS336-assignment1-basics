# from tokenizer import pretokenize
# from sortedcontainers import SortedDict

# string = "okfこん"

# byte = string.encode("utf-8")
# z = bytes([1]) + bytes([2])
# print(type(z), z)
# print(bytes([])
import cProfile
import pstats
import pickle
import re
from tokenizer import Tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  
# tokenizer = Tokenizer.from_files("/home/zizto/Alex/my_tokenizer_TineStories.pkl")
# print(tokenizer.vocab)
# print(tokenizer.merges)
text= "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
escaped_tokens = [re.escape(token) for token in special_tokens]
delimiter = "|".join(escaped_tokens)
new_pattern = f"{delimiter}|{PAT}"
print(new_pattern)
matches = re.findall(new_pattern, text)
print(matches)