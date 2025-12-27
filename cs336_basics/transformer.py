import torch
from einx import rearrange, dot, reduce, add, sum as einx_sum
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import math
import builtins
import random
import argparse
import os
import yaml
import logging
import sys
import time
from datetime import datetime
import glob
from tqdm import tqdm
import functools
import csv

def setup_logger(output_dir, verbose=True):
    """
    Setup logger that writes to both file and console.
    
    Args:
        output_dir: Directory where log file will be saved
        verbose: If True, logs to console; if False, only to file
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger('transformer_training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler - always logs to file
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only if verbose is True
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def multiply(*args):
    total = 1
    for arg in args:
        total *= arg
    return total

# TOTAL_FLOPS = 0

class Linear(torch.nn.Module): 
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros((out_features, in_features), device=device, dtype=dtype))
        std = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, 0, std, a =-3*std, b=3*std)
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_flops(x.shape)
        z = dot(
            "d_out d_in, ... d_in-> ... d_out",
            self.W, x
            )
        return z
    
    def num_flops(self, input_shape):
        # global TOTAL_FLOPS
        assert input_shape[-1] == self.W.shape[1]
        total = multiply(2, self.W.shape[0], *input_shape)
        # TOTAL_FLOPS += total
        return total

class Embedding(torch.nn.Module): 
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.embed = torch.nn.Parameter(torch.zeros((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embed, 0, 1, -3, 3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.embed[token_ids]

class RMSNorm(torch.nn.Module): 
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        result = x.to(torch.float32)
        factor = 1 / torch.sqrt(einx_sum("... [c]", (result ** 2)) / self.d_model + self.eps)
        result = dot("... c, c, ... -> ... c", result, self.gain, factor)
        return result.to(in_dtype)

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

class GLU(torch.nn.Module):
    def __init__(
        self,
        activation: torch.nn.Module, 
        linear1: Linear,
        linear2: Linear
        ):
        """
        GLU
        """
        super().__init__()
        self.activation = activation
        self.linear1 = linear1
        self.linear2 = linear2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (... d_model) and return a tensor of the same shape.
        """
        return self.activation(self.linear1(x)) * self.linear2(x)

class FFNBlock(torch.nn.Module): 
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        activation = silu,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ):
        """
        Feed Forward Layer
        """
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff -= d_ff % 64

        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)
        self.glu = GLU(activation, self.linear1, self.linear3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (... d_model) and return a tensor of the same shape.
        """
        return self.linear2(self.glu(x))

class RoPE(torch.nn.Module): 
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None 
        ):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        buffer = torch.zeros(max_seq_len, d_k // 2, 4, device=self.device)
        for pos in range(max_seq_len):
            for dim in range(self.d_k // 2):
                angle = pos / (self.theta ** (2 * dim / self.d_k))
                buffer[pos, dim, 0] = np.cos(angle)
                buffer[pos, dim, 1] = -np.sin(angle)
                buffer[pos, dim, 2] = np.sin(angle)
                buffer[pos, dim, 3] = np.cos(angle)
            
        self.register_buffer("buffer", buffer, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
        """
        sin_cos = self.buffer[token_positions]
        result = dot("... seq_len (half_d_k w1), seq_len half_d_k (w2 w1) -> ... seq_len (half_d_k w2)", x, sin_cos, w1=2)
        return result

def softmax(x: torch.Tensor, dim):
    x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x / torch.sum(x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    K : Float[Tensor, "batch_size ... seq_len d_k"],
    Q : Float[Tensor, "batch_size ... seq_len2 d_k"],
    V : Float[Tensor, "batch_size ... seq_len d_v"],
    mask : Float[Tensor, "seq_len2 seq_len"]
    ):
    QK = dot("... seq_len2 d_k, ... seq_len d_k -> ... seq_len2 seq_len", Q, K) / np.sqrt(K.shape[-1])
    mask = torch.where(mask == True, 0.0, -np.inf)
    QK = add("... seq_len2 seq_len, ... seq_len2 seq_len -> ... seq_len2 seq_len", QK, mask)
    QK = softmax(QK, -1)
    attention = dot("... seq_len2 seq_len, ... seq_len d_v -> ... seq_len2 d_v", QK, V)
    return attention

class MHSA(torch.nn.Module): 
    def __init__(
        self,
        d_model : int, 
        num_heads : int,
        rope : RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
        ):
        """
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.lin_W_o = Linear(d_model, d_model, device, dtype)
        self.lin_W_QKV = Linear(d_model, 3*d_model, device, dtype)
        self.rope = rope
        self.device = device
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        """
        seq_len = x.shape[-2]
        # print("  Linear QKV:")
        QKV = self.lin_W_QKV(x)
        Q, K, V = rearrange("... seq_len ((h + h + h) d) -> ... h seq_len d, ... h seq_len d, ... h seq_len d", QKV, h = self.num_heads)
        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(*Q.shape[:-1], seq_len, device=self.device)).to(dtype=bool)
        attention = scaled_dot_product_attention(K, Q, V, mask)
        # print("  Linear Out:")
        attention = self.lin_W_o(rearrange("... h seq_len d -> ... seq_len (h d)", attention))
        return attention

class TransformerBlock(torch.nn.Module): 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ):
        """
        Transformer Block
        """
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff -= d_ff % 64

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MHSA(d_model, num_heads, rope, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFNBlock(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (... d_model) and return a tensor of the same shape.
        """
        # print(" Attention")
        res = x + self.attn(self.norm1(x))
        # print(" FFN")
        res = res + self.ffn(self.norm2(res))
        return res

class TransformerLM(torch.nn.Module): 
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float = 10000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        ):
        """
        Transformer Block
        """
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff -= d_ff % 64

        rope = RoPE(theta, d_model // num_heads, context_length, device=device)
        self.embed = Embedding(vocab_size, d_model, device, dtype)
        blocks = []
        for i in range(num_layers):
            blocks.append(TransformerBlock(d_model, num_heads, d_ff, rope, device, dtype))
        self.blocks =  torch.nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lin_out = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (... d_model) and return a tensor of the same shape.
        """
        res = self.embed(x)
        for i, block in enumerate(self.blocks):
            # print("Block:", i + 1)
            res = block(res)
        # print("Out Linear")
        return self.lin_out(self.norm(res))

def cross_entropy_loss(logits, targets):
    logits -= torch.max(logits, dim=-1, keepdim=True).values
    sum_exp = torch.sum(torch.log(torch.sum(torch.exp(logits), dim=-1)))
    target_sum = - torch.sum(logits.gather(dim=-1, index=targets.unsqueeze(-1)))
    return (sum_exp + target_sum) / multiply(*targets.shape)

def perplexity(losses):
    return torch.exp(torch.sum(losses) / multiply(*losses.shape))

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr" : lr,
                    "betas" : betas,
                    "eps" : eps,
                    "weight_decay" : weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                first_momentum = state.get("first_momentum", torch.zeros_like(p))
                second_momentum = state.get("second_momentum", torch.zeros_like(p))
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                first_momentum = beta1 * first_momentum + (1 - beta1) * grad
                second_momentum = beta2 * second_momentum + (1 - beta2) * grad ** 2
                lr_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                p.data -= lr_t * first_momentum / (torch.sqrt(second_momentum) + eps) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data
                
                state["t"] = t + 1 # Increment iteration number.
                state["first_momentum"] = first_momentum
                state["second_momentum"] = second_momentum

        return loss

def cosine_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        a_t = t * a_max / t_w
    elif t <= t_c:
        a_t = a_min + 1/2 * (a_max - a_min) * (1 + math.cos((t - t_w) * math.pi / (t_c - t_w)))
    else:
        a_t = a_min
    return a_t

def gradient_clipping(params, max_norm, eps=1e-6):
    norm = 0.0
    for param in params:
        if param.grad is not None:
            norm += torch.sum(param.grad.data ** 2).item()
    norm = math.sqrt(norm)
    
    normalized_norm = norm
    if norm >= max_norm:
        for param in params:
            if param.grad is not None:
                param.grad.data *= max_norm / (norm + eps)
        normalized_norm = max_norm
    
    return norm, normalized_norm

def sample_batch(dataset, batch_size, context_length, device):
    x = []
    y = []
    for b in range(batch_size):
        index = random.randint(0, len(dataset) - context_length - 1)
        x.append(torch.from_numpy(dataset[index : index + context_length].copy()).to(torch.long))
        y.append(torch.from_numpy(dataset[index + 1 : index + 1 + context_length].copy()).to(torch.long))
    x = torch.stack(x, dim=0).to(torch.long).to(device)
    y = torch.stack(y, dim=0).to(torch.long).to(device)
    return (x, y)

def save_checkpoint(model, optimizer, iteration, out, loss=None):
    
    obj = {"model" : model.state_dict(),
           "optimizer" : optimizer.state_dict(),
           "iteration" : iteration
           }
    if loss is not None:
        obj["loss"] = loss
    
    torch.save(obj, out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src, map_location='cpu')
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

def train(args):
    model = TransformerLM(
        d_model = args.d_model, 
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        num_layers = args.num_layers,
        theta = args.theta,
        device = args.device,
        dtype = torch.float32
        )
    
    optimizer = AdamW(
        model.parameters(),
        lr = args.lr,
        betas = args.betas,
        eps = args.eps,
        weight_decay = args.weight_decay
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert args to dict and save as YAML
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger = setup_logger(args.output_dir, args.verbose)
    
    # Setup metrics CSV file
    metrics_file = os.path.join(args.output_dir, "metrics.csv")
    metrics_fieldnames = [
        'step', 'wallclock_time', 'elapsed_time', 'train_loss', 'train_ppl',
        'val_loss', 'val_ppl', 'learning_rate', 'grad_norm', 'normalized_grad_norm',
        'tokens_processed', 'tokens_per_sec', 'forward_time', 'backward_time', 'step_time'
    ]
    
    # Check if we're resuming and file exists
    file_exists = os.path.exists(metrics_file)
    metrics_csv = open(metrics_file, 'a', newline='')
    metrics_writer = csv.DictWriter(metrics_csv, fieldnames=metrics_fieldnames)
    if not file_exists:
        metrics_writer.writeheader()
        metrics_csv.flush()
    
    # Load datasets using memmap
    logger.info(f"Loading training data from {args.train_data_path}")
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    logger.info(f"Training data loaded: {len(train_data)} tokens")
    
    logger.info(f"Loading validation data from {args.val_data_path}")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')
    logger.info(f"Validation data loaded: {len(val_data)} tokens")
    
    # Resume from checkpoint if specified or find latest
    start_step = 0
    if args.resume_from is not None:
        # Resume from specific checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from step {start_step}")
    else:
        # Try to find latest checkpoint in output_dir
        checkpoint_pattern = os.path.join(args.output_dir, "*_step_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            # Extract step numbers and find the latest
            latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_step_')[1].split('_')[0]))
            logger.info(f"Found existing checkpoint: {latest_ckpt}")
            start_step = load_checkpoint(latest_ckpt, model, optimizer)
            logger.info(f"Resumed from step {start_step}")
    
    # Move model to device
    model = model.to(args.device)
    
    # Track total tokens processed
    total_tokens = start_step * args.batch_size * args.context_length
    
    # Track training start time
    training_start_time = time.time()
    
    logger.info("Starting training loop")
    logger.info(f"Total steps: {args.max_steps}, Starting from step: {start_step}")
    
    for step in tqdm(range(start_step, args.max_steps), initial=start_step, total=args.max_steps, desc="Training"):
        step_start_time = time.time()
        
        # Use 1-indexed steps for cosine schedule (step + 1)
        current_step = step + 1
        current_lr = cosine_schedule(
            current_step, 
            args.max_lr, 
            args.min_lr, 
            args.warmup_steps, 
            args.cooldown_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Sample batch
        x, y = sample_batch(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Gradient clipping
        grad_norm, normalized_grad_norm = gradient_clipping(model.parameters(), args.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Update tokens processed
        tokens_processed = args.batch_size * args.context_length
        total_tokens += tokens_processed
        
        step_time = time.time() - step_start_time
        tokens_per_sec = tokens_processed / step_time if step_time > 0 else 0
        
        # Logging
        if current_step % args.log_interval == 0:
            current_wallclock = time.time()
            elapsed_time = current_wallclock - training_start_time
            
            logger.info(
                f"Step: {current_step}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {torch.exp(loss).item():.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Tokens: {total_tokens} | "
                f"Tok/s: {tokens_per_sec:.0f} | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"Normalized Grad Norm: {normalized_grad_norm:.4f} | "
                f"Fwd: {forward_time:.3f}s | "
                f"Bwd: {backward_time:.3f}s | "
                f"Step: {step_time:.3f}s"
            )
            
            # Write training metrics to CSV
            metrics_writer.writerow({
                'step': current_step,
                'wallclock_time': current_wallclock,
                'elapsed_time': elapsed_time,
                'train_loss': loss.item(),
                'train_ppl': torch.exp(loss).item(),
                'val_loss': '',
                'val_ppl': '',
                'learning_rate': current_lr,
                'grad_norm': grad_norm,
                'normalized_grad_norm': normalized_grad_norm,
                'tokens_processed': total_tokens,
                'tokens_per_sec': tokens_per_sec,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'step_time': step_time
            })
            metrics_csv.flush()
        
        # Evaluation
        if current_step % args.eval_interval == 0:
            model.eval()
            eval_losses = []
            num_eval_batches = min(100, len(val_data) // (args.batch_size * args.context_length))
            
            eval_forward_start = time.time()
            with torch.no_grad():
                for _ in range(num_eval_batches):
                    x_val, y_val = sample_batch(val_data, args.batch_size, args.context_length, args.device)
                    logits_val = model(x_val)
                    loss_val = cross_entropy_loss(logits_val, y_val)
                    eval_losses.append(loss_val.item())
            eval_forward_time = time.time() - eval_forward_start
            
            avg_eval_loss = builtins.sum(eval_losses) / len(eval_losses)
            eval_perplexity = math.exp(avg_eval_loss)
            eval_tokens = num_eval_batches * args.batch_size * args.context_length
            eval_tok_per_sec = eval_tokens / eval_forward_time if eval_forward_time > 0 else 0
            
            current_wallclock = time.time()
            elapsed_time = current_wallclock - training_start_time
            
            logger.info(
                f"[EVAL] Step: {current_step} | "
                f"Val Loss: {avg_eval_loss:.4f} | "
                f"Val PPL: {eval_perplexity:.2f} | "
                f"Forward Time: {eval_forward_time:.3f}s | "
                f"Tok/s: {eval_tok_per_sec:.0f}"
            )
            
            # Write validation metrics to CSV
            metrics_writer.writerow({
                'step': current_step,
                'wallclock_time': current_wallclock,
                'elapsed_time': elapsed_time,
                'train_loss': '',
                'train_ppl': '',
                'val_loss': avg_eval_loss,
                'val_ppl': eval_perplexity,
                'learning_rate': current_lr,
                'grad_norm': '',
                'normalized_grad_norm': '',
                'tokens_processed': total_tokens,
                'tokens_per_sec': eval_tok_per_sec,
                'forward_time': eval_forward_time,
                'backward_time': '',
                'step_time': ''
            })
            metrics_csv.flush()
            
            model.train()
        
        # Checkpointing
        if current_step % args.checkpoint_interval == 0 or current_step == args.max_steps:
            date_str = datetime.now().strftime("%d_%m_%Y")
            checkpoint_name = f"{args.model_name}_step_{current_step}_date_{date_str}.pt"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            save_checkpoint(model, optimizer, current_step, checkpoint_path, loss.item())
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    metrics_csv.close()
    logger.info("Training completed!")

def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    
    new_token = None
    num_new_tokens = 0
    new_tokens = []
    cur_prompt = prompt
    while new_token != "<|endoftext|>" or num_new_tokens != max_new_tokens:
        pass
        
    
    return new_tokens

def main():
    parser = argparse.ArgumentParser(description='Train Transformer LM')
    
    # Experiment name
    parser.add_argument('--name', type=str, default='transformer_lm', help='Experiment name')
    parser.add_argument('--model_name', type=str, default='transformer', help='Model name for checkpoints')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=4*128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--theta', type=float, default=10000.0)
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-8)
    
    # Learning rate schedule
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--cooldown_steps', type=int, default=5000)
    
    # Gradient clipping
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', help='Print logs to console')
    
    
    # Paths
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    train(args)
    
if __name__ == "__main__":
    main()