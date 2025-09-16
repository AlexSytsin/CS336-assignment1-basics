import torch
from einops import rearrange, einsum, reduce
import numpy as np

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
        z = einsum(
            self.W, x,
            "d_out d_in, ... d_in-> ... d_out",
            )
        return z

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
        factor = 1 / np.sqrt(reduce((result ** 2), "... c -> ...", "sum") / self.d_model + self.eps)
        result = einsum(result, self.gain, factor, "... c, c, ... -> ... c")
        return result.to(in_dtype)
