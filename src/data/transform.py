"""Train-only preprocessing transforms."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler


class NumericTransform:
    """Clip and robust-scale numeric features."""

    def __init__(
        self,
        clip_quantiles: tuple[float, float] = (1.0, 99.0),
    ) -> None:
        self.clip_quantiles = clip_quantiles
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None
        self.scaler_: Optional[RobustScaler] = None

    def fit(self, x_num: np.ndarray) -> "NumericTransform":
        """Fit clipping bounds and the scaler on train data only."""
        if x_num.ndim != 2:
            raise ValueError("x_num must have shape [n_samples, n_features].")

        q_low, q_high = self.clip_quantiles
        if not 0.0 <= q_low < q_high <= 100.0:
            raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 100.")

        self.lower_ = np.nanpercentile(x_num, q_low, axis=0)
        self.upper_ = np.nanpercentile(x_num, q_high, axis=0)

        clipped = np.clip(x_num, self.lower_, self.upper_)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(clipped)

        return self

    def transform(self, x_num: np.ndarray) -> np.ndarray:
        """Clip and scale numeric features with fitted train statistics."""
        if self.lower_ is None or self.upper_ is None or self.scaler_ is None:
            raise RuntimeError("NumericTransform must be fit before transform().")

        if x_num.ndim != 2:
            raise ValueError("x_num must have shape [n_samples, n_features].")

        clipped = np.clip(x_num, self.lower_, self.upper_)
        return self.scaler_.transform(clipped).astype(np.float32)


class CategoricalTransform:
    """Ordinal-encode categorical input features.

    Output codes:
    - 0: padding
    - 1: unknown or missing
    - 2+: observed train categories
    """

    def __init__(self) -> None:
        self.encoder_: Optional[OrdinalEncoder] = None

    def fit(self, x_cat: np.ndarray) -> "CategoricalTransform":
        """Fit the categorical encoder on train data only."""
        if x_cat.ndim != 2:
            raise ValueError("x_cat must have shape [n_samples, n_features].")

        x_cat = self._normalize_missing(x_cat)

        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        )
        self.encoder_.fit(x_cat)

        return self

    def transform(self, x_cat: np.ndarray) -> np.ndarray:
        """Encode categorical features and reserve 0/1 for pad and unknown."""
        if self.encoder_ is None:
            raise RuntimeError("CategoricalTransform must be fit before transform().")

        if x_cat.ndim != 2:
            raise ValueError("x_cat must have shape [n_samples, n_features].")

        x_cat = self._normalize_missing(x_cat)
        encoded = self.encoder_.transform(x_cat)

        # OrdinalEncoder uses -1 for unknown/missing here.
        # Shift by +2 so:
        # -1 -> 1  (unknown/missing)
        #  0 -> 2  (first known category)
        return encoded.astype(np.int64) + 2

    @staticmethod
    def _normalize_missing(x_cat: np.ndarray) -> np.ndarray:
        """Convert missing-like values into a stable string token."""
        out = np.asarray(x_cat, dtype=object).copy()

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                value = out[i, j]

                if value is None:
                    out[i, j] = "__MISSING__"
                    continue

                try:
                    if np.isnan(value):
                        out[i, j] = "__MISSING__"
                        continue
                except TypeError:
                    pass

        return out


class TargetTransform:
    """Normalize and encode binary and family targets.

    Strategy
    - Binary:
      - numeric / bool values -> benign or malicious directly
      - string values -> infer from normalized family label
    - Family:
      - normalize text with Unicode normalization + casefold
      - map to the nearest canonical family label with TF-IDF similarity
      - final ids are produced by LabelEncoder
    """

    FAMILY_LABELS = (
        "benign",
        "bot",
        "bruteforce",
        "ddos",
        "dos",
        "infiltration",
        "scan",
        "web_attack",
        "other",
    )

    def __init__(self) -> None:
        self.binary_encoder_: Optional[LabelEncoder] = None
        self.family_encoder_: Optional[LabelEncoder] = None
        self.aux_encoders_: dict[str, Optional[LabelEncoder]] = {}

        self.family_vectorizer_: Optional[TfidfVectorizer] = None
        self.family_label_matrix_: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize text into a simple canonical form."""
        text = "" if value is None else str(value)
        text = unicodedata.normalize("NFKC", text)
        text = unicodedata.normalize("NFKC", text.casefold())
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def _normalize_family_label(self, value: Any) -> str:
        """Map a raw family label to the nearest canonical family label."""
        text = self._normalize_text(value)

        if not text:
            return "other"

        if text in self.FAMILY_LABELS:
            return text

        if self.family_vectorizer_ is None or self.family_label_matrix_ is None:
            raise RuntimeError("TargetTransform must be fit before transform().")

        query = self.family_vectorizer_.transform([text])
        scores = cosine_similarity(query, self.family_label_matrix_)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Conservative fallback for labels that are too dissimilar
        if best_score < 0.20:
            return "other"

        return self.FAMILY_LABELS[best_idx]

    def _normalize_binary_label(self, value: Any) -> str:
        """Map a raw binary label to benign or malicious."""
        if isinstance(value, (bool, np.bool_)):
            return "malicious" if bool(value) else "benign"

        if isinstance(value, (int, np.integer)):
            return "malicious" if int(value) != 0 else "benign"

        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return "malicious" if int(value) != 0 else "benign"

        family = self._normalize_family_label(value)
        return "benign" if family == "benign" else "malicious"

    def fit(
        self,
        y_binary: np.ndarray,
        y_family: np.ndarray,
        optional_aux_targets: Optional[dict[str, np.ndarray]] = None,
    ) -> "TargetTransform":
        """Fit target encoders on training targets."""
        _ = y_binary

        normalized_family_labels = [self._normalize_text(label) for label in self.FAMILY_LABELS]

        self.family_vectorizer_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            strip_accents="unicode",
            lowercase=True,
        )
        self.family_label_matrix_ = self.family_vectorizer_.fit_transform(normalized_family_labels)

        self.binary_encoder_ = LabelEncoder()
        self.binary_encoder_.fit(np.asarray(["benign", "malicious"], dtype=object))

        self.family_encoder_ = LabelEncoder()
        self.family_encoder_.fit(np.asarray(self.FAMILY_LABELS, dtype=object))

        self.aux_encoders_.clear()
        if optional_aux_targets:
            for name, values in optional_aux_targets.items():
                values = np.asarray(values)
                if values.dtype.kind in {"O", "U", "S", "b"}:
                    encoder = LabelEncoder()
                    encoder.fit(values)
                    self.aux_encoders_[name] = encoder

        return self

    def transform(
        self,
        y_binary: np.ndarray,
        y_family: np.ndarray,
        optional_aux_targets: Optional[dict[str, np.ndarray]] = None,
    ):
        """Encode binary and family targets, plus optional aux targets."""
        if self.binary_encoder_ is None or self.family_encoder_ is None:
            raise RuntimeError("TargetTransform must be fit before transform().")

        y_binary_norm = [self._normalize_binary_label(value) for value in y_binary]
        y_family_norm = [self._normalize_family_label(value) for value in y_family]

        y_binary_out = np.asarray(
            self.binary_encoder_.transform(y_binary_norm),
            dtype=np.int64,
        )
        y_family_out = np.asarray(
            self.family_encoder_.transform(y_family_norm),
            dtype=np.int64,
        )

        if optional_aux_targets is None:
            return y_binary_out, y_family_out

        aux_out: dict[str, np.ndarray] = {}
        for name, values in optional_aux_targets.items():
            values = np.asarray(values)
            encoder = self.aux_encoders_.get(name)
            aux_out[name] = encoder.transform(values) if encoder is not None else values

        return y_binary_out, y_family_out, aux_out
