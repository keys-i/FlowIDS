"""Prediction heads for the model"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    """Binary classification head for malware vs benign

    Input:
        x: [batch, in_dim]

    Output:
        logits: [batch]
    """

    def __init__(
        self,
        in_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary logits"""
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, in_dim]")

        x = self.dropout(x)
        return self.out(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """Multiclass classification head for malware family or auxiliary categorical targets

    Input:
        x: [batch, in_dim]

    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits"""
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, in_dim]")

        x = self.dropout(x)
        return self.out(x)


class RegressionHead(nn.Module):
    """Simple numeric regression head for auxiliary numeric targets

    Input:
        x: [batch, in_dim]

    Output:
        pred: [batch, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return numeric predictions"""
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, in_dim]")

        x = self.dropout(x)
        return self.out(x)


class GaussianRegressionHead(nn.Module):
    """Heteroscedastic numeric regression head

    Returns a mean and a positive variance for each target.

    Input:
        x: [batch, in_dim]

    Output:
        mean: [batch, out_dim]
        var:  [batch, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        self.mean = nn.Linear(in_dim, out_dim)
        self.log_var = nn.Linear(in_dim, out_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and positive variance"""
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, in_dim]")

        x = self.dropout(x)
        mean = self.mean(x)
        var = self.softplus(self.log_var(x)) + self.eps
        return mean, var


class Heads(nn.Module):
    """Combined heads for the flow model

    Required:
        - binary malicious vs benign head
        - family classification head

    Optional:
        - auxiliary categorical heads
        - auxiliary numeric heads
        - heteroscedastic numeric heads
    """

    def __init__(
        self,
        in_dim: int,
        num_families: int,
        dropout: float = 0.1,
        aux_class_dims: dict[str, int] | None = None,
        aux_reg_dims: dict[str, int] | None = None,
        aux_gaussian_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        self.binary = BinaryHead(
            in_dim=in_dim,
            dropout=dropout,
        )
        self.family = ClassificationHead(
            in_dim=in_dim,
            num_classes=num_families,
            dropout=dropout,
        )

        self.aux_class_heads = nn.ModuleDict(
            {
                name: ClassificationHead(
                    in_dim=in_dim,
                    num_classes=out_dim,
                    dropout=dropout,
                )
                for name, out_dim in (aux_class_dims or {}).items()
            }
        )

        self.aux_reg_heads = nn.ModuleDict(
            {
                name: RegressionHead(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                )
                for name, out_dim in (aux_reg_dims or {}).items()
            }
        )

        self.aux_gaussian_heads = nn.ModuleDict(
            {
                name: GaussianRegressionHead(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                )
                for name, out_dim in (aux_gaussian_dims or {}).items()
            }
        )

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Return all requested predictions"""
        outputs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {
            "binary_logits": self.binary(x),
            "family_logits": self.family(x),
        }

        for name, head in self.aux_class_heads.items():
            outputs[f"{name}_logits"] = head(x)

        for name, head in self.aux_reg_heads.items():
            outputs[f"{name}_pred"] = head(x)

        for name, head in self.aux_gaussian_heads.items():
            outputs[f"{name}_mean_var"] = head(x)

        return outputs
