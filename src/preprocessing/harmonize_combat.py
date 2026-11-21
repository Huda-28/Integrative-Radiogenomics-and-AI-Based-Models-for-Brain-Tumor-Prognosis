"""Phase 2: Leakage-Free Training-Fold-Only ComBat/AutoComBat Harmonization."""
from typing import Dict
import numpy as np
import pandas as pd


class FoldSafeRadiomicsHarmonizer:
    """Fits ComBat/AutoComBat parameters strictly on training folds and applies frozen parameters to validation/external folds."""

    def __init__(self, method: str = "AutoComBat", empirical_bayes: bool = True):
        self.method = method
        self.empirical_bayes = empirical_bayes
        self.site_location_: Dict[str, np.ndarray] = {}
        self.site_scale_: Dict[str, np.ndarray] = {}
        self.global_mean_: np.ndarray | None = None
        self.global_std_: np.ndarray | None = None

    def fit_training_fold(self, train_features: pd.DataFrame, train_sites: pd.Series) -> "FoldSafeRadiomicsHarmonizer":
        x = train_features.to_numpy(dtype=np.float64)
        self.global_mean_ = np.nanmean(x, axis=0)
        self.global_std_ = np.nanstd(x, axis=0) + 1e-8
        z = (x - self.global_mean_) / self.global_std_

        for site in train_sites.unique():
            mask = (train_sites == site).to_numpy()
            site_z = z[mask]
            gamma_hat = np.nanmean(site_z, axis=0)
            delta_hat = np.nanstd(site_z, axis=0) + 1e-8
            if self.empirical_bayes:
                gamma_hat = 0.85 * gamma_hat + 0.15 * np.mean(gamma_hat)
            self.site_location_[str(site)] = gamma_hat
            self.site_scale_[str(site)] = delta_hat
        return self

    def transform_fold(self, features: pd.DataFrame, sites: pd.Series) -> pd.DataFrame:
        if self.global_mean_ is None or self.global_std_ is None:
            raise RuntimeError("Fit harmonizer on the training fold before transforming validation or external cohorts.")
        if self.method == "No-Harmonization":
            return features.copy()

        x = features.to_numpy(dtype=np.float64)
        z = (x - self.global_mean_) / self.global_std_
        out = np.empty_like(z)
        for idx, site in enumerate(sites.astype(str)):
            gamma = self.site_location_.get(site, np.zeros(z.shape[1]))
            delta = self.site_scale_.get(site, np.ones(z.shape[1]))
            out[idx] = ((z[idx] - gamma) / delta) * self.global_std_ + self.global_mean_
        return pd.DataFrame(out, index=features.index, columns=features.columns)
