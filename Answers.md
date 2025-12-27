# Answers to Assignment1

## Unicode1

A. `chr(0)` represents null character, in string literal can be represented as `\0`

B. `__repr__` gives an "official" string representation, which should help to recreate the object. It means, that for example escape sequences will be printed as they are written in the code. ( `/n` for `__repr__` instead of indentation)

C. It's `__str__` returns empty string (used in `print`), but `eval()` gives `/x00`(when pasted into interpreter)  

## Unicode2

A. They typically encode strings to longer byte sequences, which is bad. For comparison `hello! こんにちは!` gives 23, 28, 56 bytes for UTF-8, UTF-16, UTF-32 respectively.

B. `hello! こんにちは!`. Because in this implementation of a function we assume one byte corresponds to one Unicode character.

C. Can take this 2 bytes (`\xe3\x81`) from 3 that encode `こ`. They don't represent any character together or by themselves.

## Train_bpe_tinystories

A. It took 29 minutes with 8 GB RAM and 6 minutes with 32 GB RAM. The longest tokens are `b' accomplishment', b' disappointment', b' responsibility'`. It makes sense, because these are pretty common long words.

B. The part that takes the longest is pretokenization (without multiprocessing). It takes 360 seconds(without multiprocessing and ~60 seconds with it), compared to 3 seconds for main tokenization loop, and a fraction of a second for chunkation.


## Train_bpe_expts_owt

A.  It took 22 minutes(with multiprocessing, 50 processes) with 32 GB RAM (and 50 GB of swap used). The longest tokens are `'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ',                            '----------------------------------------------------------------',                         '________________________________'`. 
First 10 tokens are like that, then come normal long english words 
`'telecommunications','disproportionately','environmentalists'`.
It doesn't make much sense, because these aren't some long common words opposed to tinystories.

B.  Didn't see much difference by now.

## Tokenizer Experiments

### (a) Compression Ratios

The TinyStories tokenizer (10K vocab) achieves a compression ratio of 4.194 bytes/token, while the OpenWebText tokenizer (32K vocab) achieves 4.466 bytes/token. The larger vocabulary achieves better compression as expected.

### (b) Cross-Domain Tokenization

Tokenizing OpenWebText with the TinyStories tokenizer degrades the compression ratio from 4.466 to 3.246 bytes/token (27.3% worse efficiency). The smaller vocabulary trained on simple children's stories produces 37.6% more tokens when applied to complex web text, demonstrating domain mismatch.

### (c) Tokenizer Throughput

The measured tokenization throughput is 8.74 MB/s. At this rate, tokenizing the Pile dataset (825GB) would take approximately 1.12 days (26.8 hours).

### (d) Why uint16 is Appropriate

uint16 is appropriate because both vocabularies (10K and 32K) fit comfortably within the uint16 range (0-65,535), and it uses only 2 bytes per token. This saves 50% memory compared to uint32 while ensuring no overflow for these vocabulary sizes.

## Transformer_accounting

A. 2127057600 trainable params(~2B). It would require 2127057600 * 4 = 8508230400 bytes = 8,5 GB.
<!-- 
B.  

    XL
    Block: 1
     Attention
      Linear QKV:
       15728640000 (15.7B)
      Linear Out:
       5242880000 (5B)
     FFN
       20971520000 (21B)
       20971520000 (21B)
       20971520000 (21B)
    ...
    Block: 48
     Attention
      Linear QKV:
       15728640000 (15.7B)
      Linear Out:
       5242880000 (5B)
     FFN
       20971520000 (21B)
       20971520000 (21B)
       20971520000 (21B)

    Out Linear
       164682137600 (165B)

    Total 
       4191213977600 (4.2T)

C. It's the FeedForward Blocks. They account for ~72% of all FLOPS of the model.

D. 

### Large
    Block: 1
     Attention
      Linear QKV:
       10066329600 (10B)
      Linear Out:
       3355443200 (3B)
     FFN
       16777216000 (17B)
       16777216000 (17B)
       16777216000 (17B)
    ...
    Block: 36

    Out Linear
       131745710080 (132B)

    Total 
       2426868858880 (2.4T)


### Medium
    Block: 1
     Attention
      Linear QKV:
       6442450944 (6B)
      Linear Out:
       2147483648 (2B)
     FFN
       13421772800 (13B)
       13421772800 (13B)
       13421772800 (13B)
    ...
    Block: 24

    Out Linear
       105396568064 (105B)

    Total 
       1277922639872 (1.3T)


### Small
    Block: 1
     Attention
      Linear QKV:
       3623878656 (3.6B)
      Linear Out:
       1207959552 (1.2B)
     FFN
       10066329600 (10B)
       10066329600 (10B)
       10066329600 (10B)
    ...
    Block: 12

    Out Linear
       79047426048 (79B)

    Total 
       499417350144 (500B)


Overall Attention and Last Linear projection gain more percentage, while FFN Blocks loses percentage of total FLOPS.
    

E. It becomes ~30 time more, the compute allocates a little more to attention from FFN Blocks.

    XL (Context 16384)
    Block: 1
     Attention
      Linear QKV:
       251658240000 (251B)
      Linear Out:
       83886080000 (84B)
     FFN
       335544320000 (335B)
       335544320000 (335B)
       335544320000 (335B)
    ...
    Block: 48

    Out Linear
       2634914201600 (2.6T)

    Total 
       67059424000000 (67T) -->


## Tuning the learning rate

lr = 1e1 was too slow, got from 27 to 22.

lr = 1e2 is faster, got from 22 to 3.

lr = 1e3 is the most optimal, got from 24 to 1e-23.
Because of the coincidence of tensor shape(10x10) and lr=100, we have a funny first step that just flips the sign of our weights. 
$$w_{new} = w_{old} - \frac{100}{\sqrt{1}} \cdot \left( \frac{2 w_{old}}{100} \right) = w_{old} - 2 w_{old} = -w_{old}$$

## Resource accounting for training with AdamW

### (a) Peak Memory Usage

- $V$: `vocab_size`
- $L$: `context_length`
- $N$: `num_layers`
- $d$: `d_model`
- $h$: `num_heads`
- $B$: `batch_size`
- $d_{ff} = 4d$

**1. Parameters ($M_{params}$)**
The model consists of embeddings, $L$ transformer blocks, a final normalization, and an output head.
*   **Embeddings:** $V \cdot d_{model}$
*   **Transformer Block:**
    *   RMSNorms (2): $2 \cdot d_{model}$
    *   Attention: $W_{QKV}$ ($d_{model} \times 3d_{model}$) + $W_O$ ($d_{model} \times d_{model}$) = $4 d_{model}^2$
    *   FFN (SwiGLU): 3 Linear layers ($d_{model} \times d_{ff}$). Total: $3 \cdot d_{model} \cdot d_{ff}$.
    *   Total per block: $4 d_{model}^2 + 3 d_{model} d_{ff} + 2 d_{model}$.
*   **Final Norm:** $d_{model}$
*   **Output Head:** $d_{model} \cdot V$

$$P \approx V d_{model} + L(4 d_{model}^2 + 3 d_{model} d_{ff} + 2 d_{model}) + d_{model} + V d_{model}$$
$$M_{params} = 4 \cdot P \text{ bytes}$$

**2. Gradients ($M_{grads}$)**
$$M_{grads} = 4 \cdot P \text{ bytes}$$

**3. Optimizer State ($M_{opt}$)**
AdamW stores two states (momentum and variance) per parameter.
$$M_{opt} = 2 \cdot 4 \cdot P = 8 \cdot P \text{ bytes}$$

**4. Activations ($M_{act}$)**
Based on the simplified components list provided in the prompt:
*   **Per Layer ($L$):**
    *   Input to Norm1: $B \cdot T \cdot d_{model}$
    *   Input to QKV: $B \cdot T \cdot d_{model}$
    *   Q, K, V matrices: $3 \cdot B \cdot T \cdot d_{model}$
    *   Attention Matrix (Softmax output): $B \cdot H \cdot T^2$
    *   Input to OutProj: $B \cdot T \cdot d_{model}$
    *   Input to Norm2: $B \cdot T \cdot d_{model}$
    *   Input to FFN: $B \cdot T \cdot d_{model}$
    *   FFN (Simplified 2-layer MLP per prompt):
        *   Input to SiLU (Output of W1): $B \cdot T \cdot d_{ff}$
        *   Input to W2 (Output of SiLU): $B \cdot T \cdot d_{ff}$
*   **Final:**
    *   Input to Final Norm: $B \cdot T \cdot d_{model}$
    *   Input to Output Head: $B \cdot T \cdot d_{model}$
    *   Logits: $B \cdot T \cdot V$

Summing these up:
Per Layer: $8 B T d_{model} + 2 B T d_{ff} + B H T^2$
Total Activations (bytes):
$$M_{act} = 4 \cdot B \cdot [ L(8 T d_{model} + 2 T d_{ff} + H T^2) + 2 T d_{model} + T V ]$$

**Total Peak Memory:**
$$M_{total} = 16 P + M_{act}$$

### (b) GPT-2 XL Instantiation

Using the configurations: $L=48, H=25, d_{model}=1600, T=1024, V=50257, d_{ff}=4 \times 1600 = 6400$.

1.  **Static Memory (Params + Grads + Opt):**
    *   $P \approx 2.13 \times 10^9$ parameters.
    *   Static Memory $\approx 16 \times 2.13 \text{ GB} \approx 34.03 \text{ GB}$.

2.  **Activation Memory per Batch:**
    *   Substituting values into the activation formula:
    *   $M_{act}/B \approx 10.29 \text{ GB}$.

**Expression:**
$$ \text{Memory (GB)} \approx 10.29 \cdot \text{batch\_size} + 34.03 $$

**Maximum Batch Size:**
$$ 10.29 \cdot B + 34.03 \le 80 $$
$$ 10.29 \cdot B \le 45.97 $$
$$ B \le 4.46 $$

**Maximum Batch Size:** 4

### (c) FLOPs per Step

For one step of AdamW, we perform a forward pass, a backward pass, and an optimizer update. We use the actual implementation (SwiGLU) for FLOPs calculation.

1.  **Forward Pass ($C_{fwd}$):**
    *   **MHA:** $4 B T d_{model}^2$ (projections) + $4 B T^2 d_{model}$ (attention logits & aggregation)
    *   **FFN (SwiGLU):** 3 matrices of size $d_{model} \times d_{ff}$. $3 \times 2 \times B \times T \times d_{model} \times d_{ff} = 6 B T d_{model} d_{ff}$.
    *   **Logits:** $2 B T d_{model} V$.
    *   Total per layer $\approx 4 B T d_{model}^2 + 4 B T^2 d_{model} + 6 B T d_{model} d_{ff}$.
    *   Total Forward $\approx L(4 B T d_{model}^2 + 4 B T^2 d_{model} + 6 B T d_{model} d_{ff}) + 2 B T d_{model} V$.

2.  **Backward Pass ($C_{bwd}$):**
    *   Approximated as $2 \times C_{fwd}$.

3.  **Total Compute:**
    *   $C_{step} \approx 3 \times C_{fwd}$.

**Algebraic Expression:**
$$ \text{FLOPs} \approx 3 \cdot B \cdot T \cdot [ L(4 d_{model}^2 + 4 T d_{model} + 6 d_{model} d_{ff}) + 2 d_{model} V ] $$

*Justification:* The factor of 3 accounts for the forward pass (1x) and the backward pass (2x). The term $6 d_{model} d_{ff}$ accounts for the three linear layers in the SwiGLU FFN implementation.

### (d) Training Time

*   **Total FLOPs per step (B=1024):**
    Using the formula from (c) with the GPT-2 XL parameters ($d_{ff}=6400$):
    $C_{step} \approx 1.39 \times 10^{16} \text{ FLOPs}$.
    Total Training FLOPs (400k steps) $\approx 5.55 \times 10^{21}$.

*   **Hardware Throughput:**
    A100 Peak (FP32) = 19.5 TFLOPs = $1.95 \times 10^{13}$ FLOP/s.
    Effective Throughput (50% MFU) = $9.75 \times 10^{12}$ FLOP/s.

*   **Time:**
    $$ \text{Time} = \frac{5.55 \times 10^{21}}{9.75 \times 10^{12}} \approx 5.69 \times 10^8 \text{ seconds} $$
    $$ \text{Days} = \frac{5.69 \times 10^8}{3600 \times 24} \approx 6,584 \text{ days} $$

**Answer:** ~6,584 days.