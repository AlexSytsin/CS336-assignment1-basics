





def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    merges = []
    vocab = {}
    start_tokens = [[i] for i in range(256)]
    for sym in start_tokens:
        print(bytes([0]))
        print(sym.decode("utf-8"))


    while vocab_size != len(vocab):
        break







    return vocab, merges


train_bpe(52, 300, "")