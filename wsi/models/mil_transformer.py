import math
from typing import Literal

import torch
from torch import nn
from vit_pytorch.simple_vit import Transformer as SimpleTransfomer
from vit_pytorch.vit import Transformer


def posemb_sincos_2d(x, num_grids=1, temperature=10000, dtype=torch.float32):
    _, n, dim, device, dtype = *x.shape, x.device, x.dtype
    assert dim > num_grids, "feature dimension must be larger than num_grids for 2d posemb"
    n = n//num_grids
    h, w = int(math.sqrt(n)), int(math.sqrt(n))

    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    
    # Make a tensor with the same dim as x, but with its rows being [ 1, 0, 0,...],
    # [ 0, 1, 0,...], [ 0, 0, 1,...], up to the number of grids.
    grids_base = torch.eye(num_grids, dim, device=device, dtype=dtype)

    return pe.type(dtype).repeat(1, num_grids, 1) + grids_base.repeat_interleave(n, 0)


class MilTransformer(nn.Module):
    def __init__(
        self,
        variant: Literal["vit", "simple"] = "vit",
        pos_encode: Literal["sincos", "learned", "None"] = "sincos",
        bag_size: int = 100,
        num_grids: int = 1,
        input_dim: int = 512,
        num_classes: int = 2,
        dim: int = 1024,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        self.bag_size = bag_size
        self.pos_encode = pos_encode
        self.num_grids = num_grids

        if pos_encode == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, bag_size, dim))
        elif pos_encode == "sincos":
            assert (
                bag_size == math.isqrt(bag_size) ** 2
            ), "bag_size must be a perfect square for 2d pos encode"

        self.to_token_embedding = nn.Linear(input_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        if variant == "simple":
            self.transformer = SimpleTransfomer(dim, depth, heads, dim_head, mlp_dim)
        else:
            self.transformer = Transformer(
                dim, depth, heads, dim_head, mlp_dim, dropout
            )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, bag):
        x = self.to_patch_embedding(bag)

        if self.pos_encode == "sincos":
            x += posemb_sincos_2d(x, self.num_grids)
        elif self.pos_encode == "learned":
            x += self.pos_embedding[:, : (self.bag_size + 1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)
