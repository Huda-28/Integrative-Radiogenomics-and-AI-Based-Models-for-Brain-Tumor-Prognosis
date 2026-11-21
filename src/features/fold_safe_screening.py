"""Phase 5: Nested Training-Fold-Only Feature Screening & Collinearity Control."""
from typing import List
import numpy as np
import pandas as pd


class NestedFoldFeatureScreener:
    """Prevents data leakage by screening high-dimensional habitat radiomics (wavelet/LoG) inside training folds only."""

    def __init__(self, corr_threshold: float = 0.85, max_features_per_event_ratio: float = 0.10):
        self.corr_threshold = corr_threshold
        self.max_features_per_event_ratio = max_features_per_event_ratio
        self.selected_columns_: List[str] = []

    def fit_training_fold(
        self,
        x_train: pd.DataFrame,
        survival_time: np.ndarray,
        event_observed: np.ndarray,
    ) -> "NestedFoldFeatureScreener":
        num_events = int(np.sum(event_observed == 1))
        max_k = max(8, int(num_events * self.max_features_per_event_ratio))

        # Step 1: Variance & finite check
        stds = x_train.std(axis=0)
        valid_cols = stds[stds > 1e-6].index.tolist()
        x_valid = x_train[valid_cols]

        # Step 2: Univariate concordance / correlation proxy with survival time on uncensored events
        uncensored_mask = event_observed == 1
        scores = {}
        for col in valid_cols:
            vals = x_valid.loc[uncensored_mask, col].to_numpy()
            t_vals = survival_time[uncensored_mask]
            if len(vals) > 3:
                corr = np.abs(np.corrcoef(vals, t_vals)[0, 1])
                scores[col] = float(0.0 if np.isnan(corr) else corr)
            else:
                scores[col] = 0.0

        ranked_cols = sorted(valid_cols, key=lambda c: scores[c], reverse=True)

        # Step 3: Redundancy pruning (|r| > corr_threshold)
        kept: List[str] = []
        corr_matrix = x_valid[ranked_cols].corr().abs()
        for col in ranked_cols:
            if any(corr_matrix.loc[col, k] > self.corr_threshold for k in kept):
                continue
            kept.append(col)
            if len(kept) >= max_k:
                break

        self.selected_columns_ = kept
        return self

    def transform_fold(self, x_fold: pd.DataFrame) -> pd.DataFrame:
        return x_fold[self.selected_columns_].copy()
