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

A. It took 29 minutes with 8 GB RAM and 6 minutes wirh 32 GB RAM. The longest tokens are `b' accomplishment', b' disappointment', b' responsibility'`. It makes sense, because these are pretty common long words.

B. The part that takes the longest is pretokenization (without multiprocessing). It takes 360 seconds(without multiprocessing and ~60 seconds with it), compared to 3 seconds for main tokenization loop, and a fraction of a second for chunkation.


## Train_bpe_expts_owt

A.  It took 22 minutes(with multiprocessing, 50 processes) with 32 GB RAM (and 50 GB of swap used). The longest tokens are `'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ',                            '----------------------------------------------------------------',                         '________________________________'`. 
First 10 tokens are like that, then come normal long english words 
`'telecommunications','disproportionately','environmentalists'`.
It doesn't make much sense, because these aren't some long common words opposed to tinystories.

B.  Didn't see much difference by now.