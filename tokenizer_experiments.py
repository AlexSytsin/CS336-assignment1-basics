#!/usr/bin/env python3
"""
Tokenizer experiments script for CS336 Assignment 1.
Performs experiments (a), (b), (c), and (d) as described in the problem.
"""

import numpy as np
import random
import time
from cs336_basics.tokenizer import Tokenizer, find_chunk_boundaries


# Set random seed for reproducibility
random.seed(42)

def sample_documents(filepath, n_docs=10):
    """Sample first n documents from a text file separated by <|endoftext|> token."""
    documents = []
    current_doc = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '<|endoftext|>' in line:
                # Split on the token
                parts = line.split('<|endoftext|>')
                current_doc.append(parts[0])
                
                # Save current document
                doc_text = ''.join(current_doc).strip()
                if doc_text:
                    documents.append(doc_text)
                    if len(documents) >= n_docs:
                        return documents
                
                # Start new document with remaining text after token
                current_doc = [parts[-1]] if len(parts) > 1 and parts[-1] else []
            else:
                current_doc.append(line)
    
    # Add last document if exists
    doc_text = ''.join(current_doc).strip()
    if doc_text:
        documents.append(doc_text)
    
    return documents[:n_docs]

def calculate_compression_ratio(text, tokenizer):
    """Calculate compression ratio (bytes/token) for given text and tokenizer."""
    text_bytes = len(text.encode('utf-8'))
    token_ids = tokenizer.encode(text)
    n_tokens = len(token_ids)
    
    if n_tokens == 0:
        return 0.0
    
    return text_bytes / n_tokens

def main():
    print("=" * 80)
    print("TOKENIZER EXPERIMENTS")
    print("=" * 80)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    tinystories_tokenizer = Tokenizer.from_files(
        '/Users/alex/CS336/CS336-assignment1-basics/tokenizers/TinyStoriesV2-GPT4-train.pkl',
        ["<|endoftext|>"]
    )
    owt_tokenizer = Tokenizer.from_files(
        '/Users/alex/CS336/CS336-assignment1-basics/tokenizers/owt_train.pkl',
         ["<|endoftext|>"]
    )
    print(f"TinyStories tokenizer vocab size: {len(tinystories_tokenizer.vocab)}")
    print(f"OpenWebText tokenizer vocab size: {len(owt_tokenizer.vocab)}")
    
    # ========================================================================
    # (a) Sample 10 documents and calculate compression ratios
    # ========================================================================
    print("\n" + "=" * 80)
    print("(a) COMPRESSION RATIOS ON NATIVE DATASETS")
    print("=" * 80)
    
    # Sample from TinyStories
    print("\nSampling from TinyStories...")
    ts_docs = sample_documents(
        '/Users/alex/CS336/CS336-assignment1-basics/data/raw/TinyStoriesV2-GPT4-train.txt',
        n_docs=10
    )
    ts_sample_text = '<|endoftext|>'.join(ts_docs)
    ts_compression = calculate_compression_ratio(ts_sample_text, tinystories_tokenizer)
    print(f"TinyStories sample: {len(ts_sample_text)} bytes, "
          f"{len(tinystories_tokenizer.encode(ts_sample_text))} tokens")
    print(f"TinyStories compression ratio: {ts_compression:.3f} bytes/token")
    
    # Sample from OpenWebText
    print("\nSampling from OpenWebText...")
    owt_docs = sample_documents(
        '/Users/alex/CS336/CS336-assignment1-basics/data/raw/owt_train.txt',
        n_docs=100
    )
    owt_sample_text = '<|endoftext|>'.join(owt_docs)
    owt_compression = calculate_compression_ratio(owt_sample_text, owt_tokenizer)
    print(f"OpenWebText sample: {len(owt_sample_text)} bytes, "
          f"{len(owt_tokenizer.encode(owt_sample_text))} tokens")
    print(f"OpenWebText compression ratio: {owt_compression:.3f} bytes/token")
    
    # ========================================================================
    # (b) Cross-domain tokenization: OpenWebText with TinyStories tokenizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("(b) CROSS-DOMAIN TOKENIZATION")
    print("=" * 80)
    
    print("\nTokenizing OpenWebText sample with TinyStories tokenizer...")
    owt_with_ts_compression = calculate_compression_ratio(owt_sample_text, tinystories_tokenizer)
    owt_tokens_native = len(owt_tokenizer.encode(owt_sample_text))
    owt_tokens_ts = len(tinystories_tokenizer.encode(owt_sample_text))
    
    print(f"OpenWebText with native tokenizer: {owt_tokens_native} tokens, "
          f"{owt_compression:.3f} bytes/token")
    print(f"OpenWebText with TinyStories tokenizer: {owt_tokens_ts} tokens, "
          f"{owt_with_ts_compression:.3f} bytes/token")
    print(f"Relative efficiency: {(owt_with_ts_compression / owt_compression):.2f}x worse")
    
    # Qualitative comparison
    print("\nQualitative comparison (first 100 tokens):")
    sample_text = owt_sample_text[:500]  # Take first 500 chars
    native_tokens = owt_tokenizer.encode(sample_text)
    ts_tokens = tinystories_tokenizer.encode(sample_text)
    print(f"Native tokenization: {len(native_tokens)} tokens")
    print(f"TinyStories tokenization: {len(ts_tokens)} tokens")
    print(f"Difference: {len(ts_tokens) - len(native_tokens)} more tokens "
          f"({100 * (len(ts_tokens) - len(native_tokens)) / len(native_tokens):.1f}% increase)")
    
    # ========================================================================
    # (c) Throughput estimation
    # ========================================================================
    print("\n" + "=" * 80)
    print("(c) THROUGHPUT ESTIMATION")
    print("=" * 80)
    
    # Use a representative sample for timing
    print("\nMeasuring tokenization throughput...")
    timing_text = ts_sample_text * 50000  # Repeat to get more stable timing
    timing_bytes = len(timing_text.encode('utf-8'))
    
    # Time the tokenization
    start_time = time.time()
    _ = tinystories_tokenizer.encode(timing_text)
    elapsed_time = time.time() - start_time
    
    throughput = timing_bytes / elapsed_time
    print(f"Processed {timing_bytes:,} bytes in {elapsed_time:.3f} seconds")
    print(f"Throughput: {throughput:,.0f} bytes/second ({throughput / 1024 / 1024:.2f} MB/s)")
    
    # Estimate time for Pile dataset (825GB)
    pile_bytes = 825 * 1024 * 1024 * 1024  # 825 GB in bytes
    estimated_seconds = pile_bytes / throughput
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\nEstimated time to tokenize the Pile (825GB):")
    print(f"  {estimated_seconds:,.0f} seconds")
    print(f"  {estimated_hours:,.1f} hours")
    print(f"  {estimated_days:,.2f} days")
    
    # ========================================================================
    # (d) Tokenize full datasets and save as uint16
    # ========================================================================
    print("\n" + "=" * 80)
    print("(d) TOKENIZING FULL DATASETS")
    print("=" * 80)
    
    print("\nWhy uint16 is appropriate:")
    print(f"  - TinyStories vocab size: {len(tinystories_tokenizer.vocab)} (fits in uint16: 0-65535)")
    print(f"  - OpenWebText vocab size: {len(owt_tokenizer.vocab)} (fits in uint16: 0-65535)")
    print("  - uint16 uses 2 bytes per token (vs 4 for uint32 or 8 for int64)")
    print("  - This saves 50% memory compared to uint32 for vocabularies < 65536")
    
    datasets = [
        # {
        #     'name': 'TinyStories Valid',
        #     'tokenizer': tinystories_tokenizer,
        #     'input_path': '/Users/alex/CS336/CS336-assignment1-basics/data/raw/TinyStoriesV2-GPT4-valid.txt',
        #     'output_path': '/Users/alex/CS336/CS336-assignment1-basics/data/tokenized/TinyStoriesV2-GPT4-valid.npy'
        # },
        # {
        #     'name': 'TinyStories Train',
        #     'tokenizer': tinystories_tokenizer,
        #     'input_path': '/Users/alex/CS336/CS336-assignment1-basics/data/raw/TinyStoriesV2-GPT4-train.txt',
        #     'output_path': '/Users/alex/CS336/CS336-assignment1-basics/data/tokenized/TinyStoriesV2-GPT4-train.npy'
        # },
        # {
        #     'name': 'OpenWebText Valid',
        #     'tokenizer': owt_tokenizer,
        #     'input_path': '/Users/alex/CS336/CS336-assignment1-basics/data/raw/owt_valid.txt',
        #     'output_path': '/Users/alex/CS336/CS336-assignment1-basics/data/tokenized/owt_valid.npy'
        # },
        {
            'name': 'OpenWebText Train',
            'tokenizer': owt_tokenizer,
            'input_path': '/Users/alex/CS336/CS336-assignment1-basics/data/raw/owt_train.txt',
            'output_path': '/Users/alex/CS336/CS336-assignment1-basics/data/tokenized/owt_train.npy'
        }
    ]
    
    import os

    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")

        # Get file size
        text_bytes = os.path.getsize(dataset['input_path'])
        print(f"  Input size: {text_bytes:,} bytes ({text_bytes / 1024 / 1024:.2f} MB)")

        # Use encode_file for memory-efficient tokenization
        start_time = time.time()
        dataset['tokenizer'].encode_file(
            dataset['input_path'],
            dataset['output_path'],
            chunk_size_bytes=1_000_000_000  # 1GB chunks
        )
        elapsed_time = time.time() - start_time

        # Load and get stats
        token_array = np.load(dataset['output_path'])
        total_tokens = len(token_array)
        del token_array

        print(f"  Tokenized to {total_tokens:,} tokens in {elapsed_time:.2f} seconds")
        print(f"  Throughput: {text_bytes / elapsed_time / 1024 / 1024:.2f} MB/s")
        print(f"  Compression ratio: {text_bytes / total_tokens:.3f} bytes/token")

        output_size = os.path.getsize(dataset['output_path'])
        print(f"  Saved to: {dataset['output_path']}")
        print(f"  Output size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")
        print(f"  Compression vs raw: {100 * output_size / text_bytes:.1f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF ANSWERS")
    print("=" * 80)
    
    print("\n(a) Each tokenizer's compression ratio (bytes/token):")
    print(f"    TinyStories (10K vocab): {ts_compression:.3f} bytes/token")
    print(f"    OpenWebText (32K vocab): {owt_compression:.3f} bytes/token")
    print("    The larger vocabulary achieves better compression as expected.")
    
    print("\n(b) Tokenizing OpenWebText with TinyStories tokenizer:")
    print(f"    Compression ratio degrades from {owt_compression:.3f} to {owt_with_ts_compression:.3f} bytes/token.")
    print(f"    The smaller vocabulary trained on simple children's stories produces "
          f"{100 * (owt_with_ts_compression / owt_compression - 1):.1f}% more tokens on complex web text.")
    
    print("\n(c) Tokenizer throughput:")
    print(f"    Measured throughput: {throughput / 1024 / 1024:.2f} MB/s")
    print(f"    Estimated time for Pile (825GB): {estimated_days:.2f} days")
    
    print("\n(d) Why uint16 is appropriate:")
    print(f"    Both vocabularies (10K and 32K) fit comfortably in uint16 range (0-65535),")
    print(f"    saving 50% memory compared to uint32 while ensuring no overflow.")
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
