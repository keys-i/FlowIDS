"""Train a causal flow encoder by reconstructing fields or teacher states"""

from copy import deepcopy
from typing import Self, cast, final, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.data.features import NUMERIC_COLUMNS
from src.m0.network import FlowTransformer
from src.m1.masking import CATEGORICAL_GROUPS, NUMERIC_GROUPS


@final
class RawDecoder(nn.Module):
    """Reconstruct hidden values with equally weighted feature groups"""

    def __init__(self, encoder: FlowTransformer) -> None:
        """Create numeric, missingness, and per-category output projections"""
        super().__init__()
        width = encoder.record.numeric.out_features
        self.numeric = nn.Linear(width, len(NUMERIC_COLUMNS) * 2)
        self.categorical = nn.ModuleList(
            nn.Linear(width, cast(nn.Embedding, embedding).num_embeddings)
            for embedding in encoder.record.categorical
        )

    @override
    def forward(self, hidden: Tensor, batch: dict[str, Tensor], mask: Tensor) -> Tensor:
        """Average masked losses, omitting imputed numeric values and padding

        Args:
            hidden: Student states shaped ``(batch, events, width)``
            batch: Original numeric values, missing flags, and categorical IDs
            mask: Selected groups shaped ``(batch, events, 5)``, excluding padding

        Returns:
            Scalar loss: mean numeric Smooth L1, missingness BCE, and categorical
            CE within each available group, then mean over selected groups
        """
        losses: list[Tensor] = []
        for group in range(5):
            selected = mask[..., group]
            value = hidden[selected]
            parts: list[Tensor] = []
            columns = [i for i, g in enumerate(NUMERIC_GROUPS) if g == group]
            if columns:
                numeric, missing = self.numeric(value).chunk(2, -1)
                absent = batch["missing"][selected][:, columns]
                target = batch["numeric"][selected][:, columns]
                error = F.smooth_l1_loss(numeric[:, columns], target, reduction="none")
                parts.append((error * ~absent).sum() / (~absent).sum().clamp_min(1))
                parts.append(
                    F.binary_cross_entropy_with_logits(
                        missing[:, columns], absent.float(), reduction="sum"
                    )
                    / max(absent.numel(), 1)
                )
            categorical = batch["categorical"][selected]
            category_losses = [
                F.cross_entropy(head(value), categorical[:, i], reduction="sum")
                / selected.sum().clamp_min(1)
                for i, head in enumerate(self.categorical)
                if CATEGORICAL_GROUPS[i] == group
            ]
            if category_losses:
                parts.append(torch.stack(category_losses).mean())
            losses.append(torch.stack(parts).mean())
        available = mask.any(dim=(0, 1))
        return (torch.stack(losses) * available).sum() / available.sum().clamp_min(1)


class Pretrainer(nn.Module):
    """Attach one M1 objective to the shared causal FlowTransformer"""

    def __init__(self, encoder: FlowTransformer, objective: str) -> None:
        """Build either a raw decoder or a predictor and frozen EMA teacher

        Args:
            encoder: Causal network retained after pretraining
            objective: ``reconstruct`` for fields or ``teacher`` for teacher vectors

        Raises:
            ValueError: If the objective is unknown, attention is unrestricted,
                or a latent teacher has fewer than four layers
        """
        super().__init__()
        if objective not in {"reconstruct", "teacher"} or not encoder.causal:
            raise ValueError("M1 needs a causal encoder and objective reconstruct or teacher")
        if objective == "teacher" and len(encoder.backbone.blocks) < 4:
            raise ValueError("M1 teacher needs at least four teacher layers")
        self.encoder: FlowTransformer = encoder
        self.objective: str = objective
        width = encoder.record.numeric.out_features
        self.mask_vectors: nn.Parameter = nn.Parameter(torch.zeros(5, width))
        self.decoder: RawDecoder | None = (
            RawDecoder(encoder) if objective == "reconstruct" else None
        )
        self.predictor: nn.Sequential | None = (
            nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            if objective == "teacher"
            else None
        )
        self.teacher: FlowTransformer | None = (
            deepcopy(encoder).requires_grad_(False).eval() if objective == "teacher" else None
        )

    @override
    def train(self, mode: bool = True) -> Self:
        """Set student training mode while leaving teacher dropout disabled"""
        _ = super().train(mode)
        if self.teacher is not None:
            _ = self.teacher.eval()
        return self

    def encode(self, batch: dict[str, Tensor], mask: Tensor) -> Tensor:
        """Encode the masked view without exposing hidden values or missing flags

        Args:
            batch: Numeric, missing, categorical, and left-padding tensors
            mask: Boolean group selection shaped ``(batch, events, 5)``

        Returns:
            Causal student states shaped ``(batch, events, width)``
        """
        if mask.dtype != torch.bool or mask.shape != (*batch["padding"].shape, 5):
            raise ValueError("mask must be boolean [batch, events, 5]")
        if (mask & batch["padding"][..., None]).any() or not mask.any():
            raise ValueError("mask must select valid groups and exclude padding")
        numeric_mask = mask[..., list(NUMERIC_GROUPS)]
        categorical_mask = mask[..., list(CATEGORICAL_GROUPS)]
        hidden = self.encoder.record(
            batch["numeric"].masked_fill(numeric_mask, 0),
            batch["missing"].masked_fill(numeric_mask, False),
            batch["categorical"].masked_fill(categorical_mask, 0),
        )
        hidden = hidden + mask.to(hidden.dtype) @ self.mask_vectors
        return self.encoder.backbone(hidden, batch["padding"], True)

    @override
    def forward(
        self, batch: dict[str, Tensor], mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute the selected M1 objective on masked, non-padding events

        Args:
            batch: Original unmasked features; any labels are ignored
            mask: Boolean ``(batch, events, 5)`` group selection

        Returns:
            Scalar loss, student states, and teacher states (None for reconstruct). States
            are reused for validation diagnostics without another encoder pass

        Raises:
            ValueError: If the mask has a wrong shape, selects padding, or is empty
        """
        hidden = self.encode(batch, mask)
        if self.decoder is not None:
            return self.decoder(hidden, batch, mask), hidden, None
        teacher = self.teacher
        assert teacher is not None and self.predictor is not None
        with torch.no_grad():
            target = teacher.backbone(
                teacher.record(batch["numeric"], batch["missing"], batch["categorical"]),
                batch["padding"],
                True,
                average_last=4,
            )
        selected = mask.any(-1)
        loss = F.smooth_l1_loss(self.predictor(hidden[selected]).float(), target[selected])
        return loss, hidden, target

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Move teacher parameters toward the student after a successful update

        Args:
            momentum: Retained teacher fraction, between zero and one inclusive

        Raises:
            ValueError: If momentum is outside the allowed range
        """
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0, 1]")
        if self.teacher is not None:
            for teacher, student in zip(
                self.teacher.parameters(), self.encoder.parameters(), strict=True
            ):
                _ = teacher.lerp_(student, 1 - momentum)


if __name__ == "__main__":
    from src.config import Config

    torch.set_num_threads(1)
    config = Config({"model": {"d_model": 16, "layers": 4, "heads": 2, "ffn": 32, "dropout": 0.0}})
    batch = {
        "numeric": torch.randn(2, 4, len(NUMERIC_COLUMNS)),
        "missing": torch.zeros(2, 4, len(NUMERIC_COLUMNS), dtype=torch.bool),
        "categorical": torch.ones(2, 4, len(CATEGORICAL_GROUPS), dtype=torch.long),
        "padding": torch.zeros(2, 4, dtype=torch.bool),
    }
    for objective in ("reconstruct", "teacher"):
        encoder = FlowTransformer(config, len(NUMERIC_COLUMNS), [5] * len(CATEGORICAL_GROUPS))
        model = Pretrainer(encoder, objective)
        loss, student, teacher = model(batch, torch.ones(2, 4, 5, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss) and student.shape == (2, 4, 16)
        assert encoder.record.numeric.weight.grad is not None
        assert teacher is None or (not teacher.requires_grad and teacher.shape == student.shape)
    print("M1 reconstruct/teacher forward and backward checks passed")
