"""Temporal split, train-only transforms"""

from __future__ import annotations
from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler

class RobustNumericTransform:
    """Clip, scale, and reduce numeric features

    Steps:
    1. Compute train-only median and MAD
    2. Clip to median ± clip_k * 1.4826 * MAD
    3. Scale with sklearn RobustScaler
    4. Optionally project with IncrementalPCA
    """
    def __init__(
        self,
        clip_k: float = 6.0,
        n_components: int | None = None,
        ipca_batch_sz: int | None = None,
    ) -> None:
        self.clip_k = clip_k
        self.n_components = n_components
        self.ipca_batch_sz = ipca_batch_sz

        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None
        self.scaler_: RobustScaler | None = None
        self.reducer_: IncrementalPCA | None = None


    def fit(self, x: np.ndarray) -> "RobustNumericTransform":
        """Fit clipping bounds, scaler, and reducer"""
        if x.ndim != 2:
            raise ValueError("x must have shape [n_samples, n_features].")
        
        self.median_ = np.median(x, axis=0)
        self.mad_ = np.median(np.abs(x - self.median_), axis=0)

        sigma = 1.4826 * np.where(self.mad_ == 0.0, 1.0, self.mad_)
        self.lower_ = self.median_ - self.clip_k * sigma
        self.upper_ = self.median_ + self.clip_k * sigma

        clipped = np.clip(x, self.lower_, self.upper_)
        self.scaler_ = RobustScaler()
        scaled = self.scaler_.fit_transform(clipped)

        if self.n_components is not None:
            n_features = scaled.shape[1]
            n_components = min(self.n_components, n_features)
            self.reducer_ = IncrementalPCA()
        pass