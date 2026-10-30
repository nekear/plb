"""Set Transformer building blocks (Lee et al., 2019, ICML).

Vendored from https://github.com/juho-lee/set_transformer (MIT License,
Copyright (c) 2019 Juho Lee). Modifications for this project:

- ISAB dropped: neighbor sets here are at most ~90 elements, so the quadratic
  self-attention of SAB is cheap and inducing points buy nothing;
- MAB/SAB/PMA gained an optional ``key_padding_mask`` argument (bool ``[B, N_k]``,
  True marks a padded slot, matching the torch convention) so padded neighbor slots
  cannot receive attention.

The padding fill is a large finite negative rather than ``-inf``: a row whose keys
are ALL padded would otherwise softmax to NaN and poison gradients. With the finite
fill such rows produce uniform garbage attention instead, and the model zeroes their
pooled vector downstream, so no gradient flows through them.
"""

# ruff: noqa: E741  (the vendored code names the attention output `O`; kept verbatim)

from __future__ import annotations

import math

import torch
import torch.nn as nn

_MASK_FILL = -1e9


class MAB(nn.Module):
    def __init__(
        self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False
    ) -> None:
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if key_padding_mask is not None:
            pad = key_padding_mask.repeat(self.num_heads, 1).unsqueeze(
                1
            )  # [B*h, 1, N_k]
            scores = scores.masked_fill(pad, _MASK_FILL)
        A = torch.softmax(scores, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + torch.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False
    ) -> None:
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(
        self, X: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.mab(X, X, key_padding_mask)


class PMA(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_seeds: int, ln: bool = False
    ) -> None:
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(
        self, X: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, key_padding_mask)
