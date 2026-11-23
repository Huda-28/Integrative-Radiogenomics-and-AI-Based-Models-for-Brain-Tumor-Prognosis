"""Phases 8 & 9: Triadic Survival Validation & Quantitative XAI Insertion/Deletion Faithfulness."""
import numpy as np


def harrell_concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Computes Harrell's C-index for right-censored survival predictions."""
    concordant = 0.0
    permissible = 0.0
    n = len(event_times)
    for i in range(n):
        if event_observed[i] != 1:
            continue
        for j in range(n):
            if event_times[j] > event_times[i]:
                permissible += 1.0
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1.0
                elif predicted_risk[i] == predicted_risk[j]:
                    concordant += 0.5
    return float(concordant / max(1.0, permissible))


def calibration_slope_and_mace(
    observed_survival_12m: np.ndarray,
    predicted_survival_12m: np.ndarray,
) -> tuple[float, float]:
    """Computes Calibration Slope (ideal = 1.0) and Mean Absolute Calibration Error (MACE)."""
    pred = np.clip(predicted_survival_12m, 1e-4, 1.0 - 1e-4)
    obs = observed_survival_12m.astype(np.float64)
    slope = float(np.cov(pred, obs)[0, 1] / max(1e-8, np.var(pred)))
    mace = float(np.mean(np.abs(pred - obs)))
    return slope, mace


def xai_deletion_faithfulness_score(
    feature_matrix: np.ndarray,
    shap_importance_order: np.ndarray,
    baseline_risk_scores: np.ndarray,
    predict_fn,
) -> float:
    """
    Addresses Survey Section 6.4 & Challenge 5:
    Quantifies XAI faithfulness by progressively zeroing top-ranked SHAP features and
    measuring monotonic correlation between attribution rank and risk perturbation.
    """
    n, d = feature_matrix.shape
    perturbed = feature_matrix.copy()
    drops = []
    for k in range(min(d, 10)):
        col_idx = int(shap_importance_order[k])
        perturbed[:, col_idx] = 0.0
        new_risk = predict_fn(perturbed)
        drops.append(float(np.mean(np.abs(baseline_risk_scores - new_risk))))
    if len(drops) < 2:
        return 1.0
    ranks = np.arange(len(drops), dtype=np.float64)
    corr = float(np.corrcoef(ranks, np.asarray(drops))[0, 1])
    return float(np.clip(0.5 + 0.5 * (0.0 if np.isnan(corr) else corr), 0.0, 1.0))
