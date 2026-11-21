"""Phase 2: Multicenter MRI Intensity Standardization & ComBat/AutoComBat Harmonization."""
from typing import Dict, List
import numpy as np
import pandas as pd


class MulticenterRadiomicsHarmonizer:
    """Harmonizes radiomic feature distributions across scanner sites and field strengths (1.5T vs 3.0T)."""

    def __init__(self, method: str = "AutoComBat", empirical_bayes: bool = True):
        self.method = method
        self.empirical_bayes = empirical_bayes
        self.site_Location_: Dict[str, np.ndarray] = {}
        self.site_scale_: Dict[str, np.ndarray] = {}
        self.global_mean_: np.ndarray | None = None
        self.global_std_: np.ndarray | None = None

    def fit(self, features: pd.DataFrame, batch_site: pd.Series, biological_covariates: pd.DataFrame) -> "MulticenterRadiomicsHarmonizer":
        x = features.to_numpy(dtype=np.float64)
        self.global_mean_ = np.nanmean(x, axis=0)
        self.global_std_ = np.nanstd(x, axis=0) + 1e-8
        z = (x - self.global_mean_) / self.global_std_

        for site in batch_site.unique():
            mask = (batch_site == site).to_numpy()
            site_z = z[mask]
            gamma_hat = np.nanmean(site_z, axis=0)
            delta_hat = np.nanstd(site_z, axis=0) + 1e-8
            if self.empirical_bayes:
                prior_mean = np.mean(gamma_hat)
                gamma_hat = 0.85 * gamma_hat + 0.15 * prior_mean
            self.site_Location_[str(site)] = gamma_hat
            self.site_scale_[str(site)] = delta_hat
        return self

    def transform(self, features: pd.DataFrame, batch_site: pd.Series) -> pd.DataFrame:
        if self.global_mean_ is None or self.global_std_ is None:
            raise RuntimeError("Harmonizer must be fitted on discovery cohorts before transform.")
        x = features.to_numpy(dtype=np.float64)
        z = (x - self.global_mean_) / self.global_std_
        out = np.empty_like(z)

        for idx, site in enumerate(batch_site.astype(str)):
            gamma = self.site_Location_.get(site, np.zeros(z.shape[1]))
            delta = self.site_scale_.get(site, np.ones(z.shape[1]))
            out[idx] = ((z[idx] - gamma) / delta) * self.global_std_ + self.global_mean_

        return pd.DataFrame(out, index=features.index, columns=features.columns)
