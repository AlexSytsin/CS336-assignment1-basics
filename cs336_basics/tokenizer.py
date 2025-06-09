import regex as re
import os
from typing import BinaryIO
from sortedcontainers import SortedSet

class Pretoken:
    def __init__(self, tokens: bytes):
        self.tokens = tokens

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(
    text: str,
    special_tokens: list[str],
) -> dict[bytes, int]:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = {}
    # split the chunk on all special tokens(using re.escape to "экранировать" all "|" in special tokens)
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = "|".join(escaped_tokens)
    chunks_no_special_tokens = re.split(delimiter, text)
    for chunk in chunks_no_special_tokens:
        pretokens = re.findall(PAT, chunk)
        iter = re.finditer(PAT, chunk)
        for pretoken in iter:
            pretoken_bytes = pretoken.group().encode("utf-8")
            key_bytes = []
            for byte in pretoken_bytes:
                key_bytes.append(bytes([byte]))
            key_bytes = tuple(key_bytes)
            counts[key_bytes] = counts.get(key_bytes, 0) + 1
    return counts

def delete_from_pair_counts_sorted(pair, map_pair_to_count, pair_counts_sorted):
    cur_count = map_pair_to_count.get(pair, 0)
    pair_counts_sorted.discard((cur_count, pair[0], pair[1]))

def add_to_pair_counts_sorted(pair, map_pair_to_count, pair_counts_sorted):
    cur_count = map_pair_to_count[pair]
    pair_counts_sorted.add((cur_count, pair[0], pair[1]))

def update_pairs(old_pair, cur_pair, new_pair, pretoken_count, map_pair_to_count, pair_counts_sorted):

    delete_from_pair_counts_sorted(old_pair, map_pair_to_count, pair_counts_sorted)
    map_pair_to_count[old_pair] -= pretoken_count
    add_to_pair_counts_sorted(old_pair, map_pair_to_count, pair_counts_sorted)

    delete_from_pair_counts_sorted(new_pair, map_pair_to_count, pair_counts_sorted)
    map_pair_to_count[new_pair] = map_pair_to_count.get(new_pair, 0) + pretoken_count            
    add_to_pair_counts_sorted(new_pair, map_pair_to_count, pair_counts_sorted)

    delete_from_pair_counts_sorted(cur_pair, map_pair_to_count, pair_counts_sorted)
    map_pair_to_count[cur_pair] -= pretoken_count
    add_to_pair_counts_sorted(cur_pair, map_pair_to_count, pair_counts_sorted)

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Initializing vocab with 256 bytes and special tokens
    merges = []
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i]) 
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # chunk and pretokenize
    split_special_token = "<|endoftext|>".encode("utf-8")
    pretoken_counts = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            local_pretoken_counts = pretokenize(chunk, special_tokens)
            for pretoken in local_pretoken_counts:
                pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + local_pretoken_counts[pretoken]

    # token pair counts
    pair_counts_sorted = SortedSet() # sorted set of tuples (count, tok1, tok2) 
    map_pair_to_pretokens = dict()
    map_pair_to_count = dict()
    map_pretoken_to_class = dict()
    for pretoken, count in pretoken_counts.items():
        for tok1, tok2 in zip(pretoken[:-1],pretoken[1:]):
            if pretoken not in map_pretoken_to_class:
                map_pretoken_to_class[pretoken] = Pretoken(pretoken)
            pretoken_link = map_pretoken_to_class[pretoken]
            map_pair_to_pretokens.setdefault((tok1, tok2), set()).add(pretoken_link)
            map_pair_to_count[(tok1,tok2)] = map_pair_to_count.get((tok1,tok2), 0) + count
    for tok_pair, count in map_pair_to_count.items():
        pair_counts_sorted.add((count, tok_pair[0], tok_pair[1]))

    # main tokenization loop
    while len(vocab) != vocab_size:
        top_pair = pair_counts_sorted.pop()
        pair_counts_sorted.add(top_pair)
        count = top_pair[0]
        top_pair = top_pair[1:]
        merges.append(top_pair)
        vocab[len(vocab)] = top_pair[0] + top_pair[1]

        pretokens = map_pair_to_pretokens[top_pair]
        for pretoken_link in pretokens:
            new_pretoken = []
            pretoken = pretoken_link.tokens
            # if pretoken not in pretoken_counts:
            #     continue
            pretoken_count = pretoken_counts[pretoken]
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == top_pair:
                    # tok1, tok2, tok3(merging tok2 and tok3) erase (tok1, tok2) pair and add (tok1, tok2tok3) pair 
                    if i != 0:
                        old_pair = (new_pretoken[-1], pretoken[i])
                        new_pair = (new_pretoken[-1], pretoken[i] + pretoken[i + 1])
                        update_pairs(old_pair, top_pair, new_pair, pretoken_count, map_pair_to_count, pair_counts_sorted)

                    # tok1, tok2, tok3(merging tok1 and tok2) erase (tok2, tok3) pair and add (tok1tok2, tok3) pair
                    if i != len(pretoken) - 2:
                        old_pair = (pretoken[i + 1], pretoken[i + 2])
                        new_pair = (pretoken[i] + pretoken[i + 1], pretoken[i + 2])
                        update_pairs(old_pair, top_pair, new_pair, pretoken_count, map_pair_to_count, pair_counts_sorted)

                    new_pretoken.append(pretoken[i] + pretoken[i + 1])
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            
            new_pretoken = tuple(new_pretoken)
            if new_pretoken != pretoken:
                pretoken_link.tokens = new_pretoken
                for tok1, tok2 in zip(new_pretoken[:-1], new_pretoken[1:]):
                    map_pair_to_pretokens.setdefault((tok1, tok2), set()).add(pretoken_link)
                pretoken_counts[new_pretoken] = pretoken_counts.get(new_pretoken, 0) + pretoken_counts[pretoken]
                del pretoken_counts[pretoken]
        pair_counts_sorted.discard((map_pair_to_count[top_pair],top_pair[0],top_pair[1]))
        del map_pair_to_pretokens[top_pair]
        del map_pair_to_count[top_pair]
    return vocab, merges

# vocab, merges = train_bpe("/home/zizto/Alex/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
# vocab, merges = train_bpe("/home/zizto/Alex/CS336/assignment1-basics/tests/fixtures/corpus.en", 500, ["<|endoftext|>"])