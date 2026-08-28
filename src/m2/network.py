"""Hybrid and future-only objectives with a shared EMA teacher"""

from typing import final, override

import torch
from torch import Tensor, nn

from src.m0.network import FlowTransformer
from src.m1.network import Pretrainer, RawDecoder


@final
class M2Pretrainer(Pretrainer):
    """Train a hybrid objective or future-only JEPA using M1 components"""

    def __init__(self, encoder: FlowTransformer, objective: str) -> None:
        """Select raw/current/future losses and create the required heads"""
        if objective not in {"hybrid", "future-hybrid", "future-jepa"}:
            raise ValueError("M2 objective must be hybrid, future-hybrid, or future-jepa")
        super().__init__(encoder, "teacher")
        self.objective = objective
        self.raw_decoder = RawDecoder(encoder) if objective != "future-jepa" else None
        if objective == "future-jepa":
            self.predictor = None
        width = encoder.record.numeric.out_features
        self.future_predictor = (
            nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            if objective.startswith("future-")
            else None
        )
        self.horizon = (
            nn.Parameter(torch.zeros(3, width)) if self.future_predictor is not None else None
        )
        self.loss_names = (
            ("future",)
            if objective == "future-jepa"
            else ("raw", "latent", "future")
            if self.future_predictor is not None
            else ("raw", "latent")
        )
        self.register_buffer("loss_weights", torch.ones(len(self.loss_names)))
        self.loss_weights: Tensor = self.get_buffer("loss_weights")

    def losses(self, batch: dict[str, Tensor], mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return unweighted losses and reusable student/teacher states

        Future histories are encoded only by the frozen teacher. Missing horizons
        are omitted, then horizon losses are averaged within each eligible anchor
        """
        teacher = None
        losses: list[Tensor] = []
        if self.raw_decoder is None:
            student = self.encode(batch, mask)
        else:
            latent, student, teacher = super().forward(batch, mask)
            losses.extend((self.raw_decoder(student, batch, mask), latent))
        if self.future_predictor is not None:
            assert self.teacher is not None and self.horizon is not None
            indices = batch["future_index"]
            valid = indices >= 0
            if indices.shape != (len(mask), 3) or not valid.any(1).all():
                raise ValueError("each future anchor needs at least one of three horizons")
            with torch.no_grad():
                future_states = self.teacher.backbone(
                    self.teacher.record(
                        batch["future_numeric"],
                        batch["future_missing"],
                        batch["future_categorical"],
                    ),
                    batch["future_padding"],
                    True,
                    average_last=4,
                )
                targets = future_states[:, -1][indices.clamp_min(0)]
                if teacher is None:
                    teacher = future_states
            prediction = self.future_predictor(student[:, -1, None] + self.horizon).float()
            prediction = prediction / (prediction.norm(dim=-1, keepdim=True) + 1e-6)
            targets = targets / (targets.norm(dim=-1, keepdim=True) + 1e-6)
            error = (prediction - targets).square().sum(-1)
            losses.append(((error * valid).sum(1) / valid.sum(1)).mean())
        assert teacher is not None
        return torch.stack(losses), student, teacher

    @override
    def forward(self, batch: dict[str, Tensor], mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply fixed loss weights; hybrid weights come from training-only warm-up"""
        losses, student, teacher = self.losses(batch, mask)
        return (losses * self.loss_weights).sum(), student, teacher


if __name__ == "__main__":
    from src.config import Config
    from src.data.features import NUMERIC_COLUMNS
    from src.m1.masking import CATEGORICAL_GROUPS

    torch.set_num_threads(1)
    config = Config({"model": {"d_model": 16, "layers": 4, "heads": 2, "ffn": 32, "dropout": 0.0}})
    batch = {
        "numeric": torch.randn(2, 4, len(NUMERIC_COLUMNS)),
        "missing": torch.zeros(2, 4, len(NUMERIC_COLUMNS), dtype=torch.bool),
        "categorical": torch.ones(2, 4, len(CATEGORICAL_GROUPS), dtype=torch.long),
        "padding": torch.zeros(2, 4, dtype=torch.bool),
    }
    batch.update({f"future_{key}": value.clone() for key, value in list(batch.items())})
    batch["future_index"] = torch.tensor([[0, 1, -1], [1, -1, -1]])
    for name in ("hybrid", "future-hybrid", "future-jepa"):
        encoder = FlowTransformer(config, len(NUMERIC_COLUMNS), [5] * len(CATEGORICAL_GROUPS))
        model = M2Pretrainer(encoder, name)
        loss, _, teacher = model(batch, torch.ones(2, 4, 5, dtype=torch.bool))
        loss.backward()
        assert torch.isfinite(loss) and not teacher.requires_grad
        assert encoder.record.numeric.weight.grad is not None
        if name == "future-jepa":
            assert model.raw_decoder is None and model.predictor is None
    print("M2 hybrid/future-hybrid/future-jepa forward and backward checks passed")
