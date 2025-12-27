from transformer import TransformerLM, Linear, TOTAL_FLOPS, multiply
import torch


vocab_size = 50257
context_length = 16384 # 1024
num_layers = 1 # S - 12 # M - 24 # L - 36 # XL - 48 
d_model = 1600 # S - 768 # M - 1024 # L - 1280 # XL - 1600
num_heads = 25 # S - 12 # M - 16 # L - 20 # XL - 25
d_ff = 6400 #XL - 6400
device = torch.device("mps")

lm = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers)
input = torch.ones(context_length, d_model, dtype=torch.int)
# lm(input)
from transformer import TOTAL_FLOPS
print("Total Flops: ", TOTAL_FLOPS)

TOTAL_PARAMS = 0
# for name, parameter in lm.named_parameters():
#     # print(name) 
#     TOTAL_PARAMS += multiply(*parameter.shape)

# print("Total Params: ", TOTAL_PARAMS) 



lin_QKV = Linear(d_model, 3 * d_model)
lin_o = Linear(d_model, d_model)
print(lin_QKV.num_flops((context_length, d_model)))
print(lin_o.num_flops((context_length, d_model)))