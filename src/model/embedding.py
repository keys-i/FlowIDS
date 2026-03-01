"""Embedding blocks for the model"""

from __future__ import annotations

import torch
import torch.nn as nn


class NumericProjection(nn.Module):
    """Project numeric features to the embedding dimension"""

    def __init__(
        self,
        in_features: int,
        embedding_dim: int,
        bias: bool = True,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.embedding_dim = embedding_dim

        self.proj = nn.Linear(in_features, embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """Project numeric inputs.

        Args:
            x_num: Tensor of shape [batch, seq_len, num_numeric].

        Returns:
            Tensor of shape [batch, seq_len, embedding_dim].
        """
        if x_num.ndim != 3:
            raise ValueError("x_num must have shape [batch, seq_len, num_numeric].")

        if x_num.size(-1) != self.in_features:
            raise ValueError(f"Expected x_num last dim {self.in_features}, got {x_num.size(-1)}.")

        x = self.proj(x_num)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class CategoricalEmbedding(nn.Module):
    """Embed categorical timestep features into model space.

    Each categorical feature gets its own embedding table.
    The embeddings are summed to produce a single tensor of shape
    [batch, seq_len, embedding_dim].
    """

    def __init__(
        self,
        cardinalities: dict[str, int],
        embedding_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        if not cardinalities:
            raise ValueError("cardinalities must not be empty.")

        self.feature_names = list(cardinalities.keys())
        self.num_categorical = len(self.feature_names)
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
                for name, cardinality in cardinalities.items()
            }
        )

        self.norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Embed categorical inputs.

        Args:
            x_cat: Tensor of shape [batch, seq_len, num_categorical].

        Returns:
            Tensor of shape [batch, seq_len, embedding_dim].
        """
        if x_cat.ndim != 3:
            raise ValueError("x_cat must have shape [batch, seq_len, num_categorical].")

        if x_cat.size(-1) != self.num_categorical:
            raise ValueError(
                f"Expected x_cat last dim {self.num_categorical}, got {x_cat.size(-1)}."
            )

        x = None

        for i, name in enumerate(self.feature_names):
            emb = self.embeddings[name](x_cat[..., i])
            x = emb if x is None else x + emb

        x = self.norm(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Fuse numeric projection and categorical embeddings.

    Inputs:
        x_num: [batch, seq_len, num_numeric]
        x_cat: [batch, seq_len, num_categorical]

    Output:
        x: [batch, seq_len, embedding_dim]
    """

    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: dict[str, int],
        embedding_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.numeric = NumericProjection(
            in_features=num_numeric,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_layer_norm=False,
        )
        self.categorical = CategoricalEmbedding(
            cardinalities=categorical_cardinalities,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_layer_norm=False,
        )

        self.norm = nn.LayerNorm(embedding_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse numeric and categorical timestep representations."""
        x = self.numeric(x_num) + self.categorical(x_cat)
        x = self.norm(x)
        x = self.dropout(x)
        return x
