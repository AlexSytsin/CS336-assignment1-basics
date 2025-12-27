import regex as re
import os
from typing import BinaryIO, Iterable, Iterator
from sortedcontainers import SortedSet
import pickle
import time
import multiprocessing
import argparse
import numpy as np
import tempfile

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)
    

class Tokenizer:
    def __init__(
        self,
        vocab:  dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
        ):
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """

        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
            special_tokens = set(special_tokens)
        else:
            special_tokens = set()
        self.special_tokens = special_tokens
        self.reversed_vocab = {value : key for key, value in self.vocab.items()}
        
        # Build merge lookup for O(1) access and priority ordering
        self.merge_priority = {pair: idx for idx, pair in enumerate(self.merges)}

    @classmethod
    def from_files(
        cls,
        tokenizer_filepath: str,
        special_tokens: list[str] | None = None
        ): 
        """Class
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """

        vocab, merges = deserialize_with_pickle(tokenizer_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode an input text into a sequence of token IDs.
        
        Args:
            text: Input text to encode
        
        Returns:
            np.ndarray of token IDs (dtype=uint16)
        """
        # Initialize cache if not already present (for encode_file)
        if not hasattr(self, 'pretoken_cache'):
            self.pretoken_cache = {}
        
        # Handle special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, reverse=True, key=len)]
            delimiter = "|".join(escaped_tokens)
            chunks_and_special_tokens = re.split(f"({delimiter})", text)
        else:
            chunks_and_special_tokens = [text]
        
        # Collect numpy arrays to concatenate at the end
        answer_arrays = []
        for chunk in chunks_and_special_tokens:
            if self.special_tokens and chunk in self.special_tokens:
                # Special token - create single element array
                token_id = self.reversed_vocab[chunk.encode("utf-8")]
                answer_arrays.append(np.array([token_id], dtype=np.uint16))
            else:
                # Use pre-compiled regex pattern
                for pretoken_obj in COMPILED_PAT.finditer(chunk):
                    pretoken = pretoken_obj.group()
                    pretoken_bytes = pretoken.encode("utf-8")
                    cached = self.pretoken_cache.get(pretoken_bytes)
                    if cached is None:
                        cached = self._merge_pretoken(pretoken_bytes)
                        self.pretoken_cache[pretoken_bytes] = cached
                    answer_arrays.append(cached)
        
        # Concatenate all arrays efficiently
        if not answer_arrays:
            return np.array([], dtype=np.uint16).tolist()
        return np.concatenate(answer_arrays).tolist()
    
    def encode_normal(self, text: str) -> list[int]:
        return self.encode(text).tolist()
  
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: 
        """Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.
        """

        for str in iterable:
            token_ids = self.encode(str)
            for id in token_ids:
                yield int(id)

    def decode(self, ids: list[int] | np.ndarray) -> str: 
        """
        Decode a sequence of token IDs into text.
        """
        answ = b''
        for id in ids:
            answ += self.vocab[int(id)]
        return answ.decode('utf-8', 'replace')
    
    def encode_file(self, file_path: str, output_path: str, chunk_size_bytes: int = 1_000_000_000) -> None:
        """
        Encode a large file into token IDs and save to a numpy file.
        Uses file-based chunking to avoid memory overflow.
        
        Args:
            file_path: Path to input text file
            output_path: Path to save tokenized output (.npy)
            chunk_size_bytes: Target size of each text chunk in bytes (default 1GB)
        """
        # Get file size and calculate chunk boundaries
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            
            if file_size == 0:
                np.save(output_path, np.array([], dtype=np.uint16))
                return
            
            desired_num_chunks = max(1, file_size // chunk_size_bytes)
            split_special_token = "<|endoftext|>".encode("utf-8")
            boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)
        
        # Initialize persistent cache for all chunks
        self.pretoken_cache: dict[bytes, np.ndarray] = {}
        
        # Process each chunk and save to temp files
        temp_files = []
        chunk_token_counts = []
        
        try:
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                if start == end:
                    continue
                
                # Read chunk
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    chunk_bytes = f.read(end - start)
                    if not chunk_bytes:
                        continue
                    chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
                
                # Encode chunk (uses persistent cache)
                chunk_tokens = self.encode(chunk_text)
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.npy')
                np.save(temp_file, chunk_tokens)
                temp_file.close()
                
                temp_files.append(temp_file.name)
                chunk_token_counts.append(len(chunk_tokens))
            
            # Calculate total token count
            total_tokens = sum(chunk_token_counts)
            
            if total_tokens == 0:
                np.save(output_path, np.array([], dtype=np.uint16))
                return
            
            # Write chunks directly to output file sequentially
            # This avoids loading everything into memory
            with open(output_path, 'wb') as out_f:
                # Write numpy .npy header manually
                header = {
                    'descr': np.dtype(np.uint16).descr[0][1],
                    'fortran_order': False,
                    'shape': (total_tokens,)
                }
                np.lib.format.write_array_header_2_0(out_f, header)
                
                # Append each chunk's data
                for temp_file_path in temp_files:
                    chunk_data = np.load(temp_file_path)
                    chunk_data.tofile(out_f)
                    del chunk_data  # Free immediately
        
        finally:
            # Clean up temp files
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            
            # Clean up cache
            if hasattr(self, 'pretoken_cache'):
                del self.pretoken_cache
    
    def _merge_pretoken(self, pretoken_bytes: bytes) -> np.ndarray:
        """
        Apply BPE merges to a single pretoken.
        
        Args:
            pretoken_bytes: Byte representation of pretoken
        
        Returns:
            np.ndarray of token IDs after merging (dtype=uint16)
        """
        cur = [bytes([byte]) for byte in pretoken_bytes]

        while len(cur) > 1:
            best_idx = -1
            best_priority = len(self.merge_priority)
            for i in range(len(cur) - 1):
                pair = (cur[i], cur[i + 1])
                priority = self.merge_priority.get(pair)
                if priority is None:
                    continue
                if priority < best_priority:
                    best_priority = priority
                    best_idx = i
            if best_idx == -1:
                break
            cur = cur[:best_idx] + [cur[best_idx] + cur[best_idx + 1]] + cur[best_idx + 2:]

        return np.array([self.reversed_vocab[token_bytes] for token_bytes in cur], dtype=np.uint16)

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

    mini_chunk_size = 131072  # Read ahead by 4k bytes at a time

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
    
    counts = {}
    # split the chunk on all special tokens(using re.escape to "экранировать" all "|" in special tokens)
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = "|".join(escaped_tokens)
    chunks_no_special_tokens = re.split(delimiter, text)
    for chunk in chunks_no_special_tokens:
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
    start_time = time.time()

    split_special_token = "<|endoftext|>".encode("utf-8")
    pretoken_counts = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        end_time = time.time()
        print(f"-> Time for Chunkation: {end_time - start_time:.4f} seconds")

        start_time = time.time()
        args_for_processes = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args_for_processes.append((chunk, special_tokens))

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        list_of_counts = pool.starmap(pretokenize, args_for_processes)

    for local_pretoken_counts in list_of_counts:
        for pretoken in local_pretoken_counts:
            pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + local_pretoken_counts[pretoken]
    
    end_time = time.time()
    print(f"-> Time for Pretokenization: {end_time - start_time:.4f} seconds")



    # token pair counts
    start_time = time.time()

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

    end_time = time.time()
    print(f"-> Time for Token Pair Counting: {end_time - start_time:.4f} seconds")


    # main tokenization loop
    start_time = time.time()

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

    end_time = time.time()
    print(f"-> Time for Main Tokenization: {end_time - start_time:.4f} seconds")

    return vocab, merges

def serialize_with_pickle(vocab, merges, filename):
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    model = {"vocab": vocab, "merges": merges}
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def deserialize_with_pickle(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model['vocab'], model['merges']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script that greets the user.")
    parser.add_argument("--dataset", type=str, required=True, help="The path to dataset")
    parser.add_argument("--vocab-size", type=int, required=True, help="The size of learnt vocabulary")
    parser.add_argument("--num-processes", type=int, required=True, help="The number of processes spawn during pretokenization step")
    parser.add_argument("--output-file", type=str, required=True, help="The path where to store learnt tokenizer")

    args = parser.parse_args()

    vocab, merges = train_bpe(args.dataset, args.vocab_size, ["<|endoftext|>"], args.num_processes) # "data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], 10
    serialize_with_pickle(vocab, merges, args.output_file)
