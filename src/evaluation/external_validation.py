"""Phase 9: Nested CV & Locked External Validation (C-Index, Calibration Slope, MACE, DCA & Paired Bootstrap)."""
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


def paired_bootstrap_c_index_test(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    risk_model_a: np.ndarray,
    risk_model_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap resampling test comparing multimodal fusion (Model A) against baseline (Model B)."""
    rng = np.random.default_rng(seed)
    n = len(event_times)
    deltas = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        c_a = harrell_concordance_index(event_times[idx], risk_model_a[idx], event_observed[idx])
        c_b = harrell_concordance_index(event_times[idx], risk_model_b[idx], event_observed[idx])
        deltas.append(c_a - c_b)
    deltas_arr = np.asarray(deltas)
    p_val = float(np.mean(deltas_arr <= 0.0))
    return {
        "mean_delta_c_index": float(np.mean(deltas_arr)),
        "ci95_low": float(np.percentile(deltas_arr, 2.5)),
        "ci95_high": float(np.percentile(deltas_arr, 97.5)),
        "p_value": p_val,
    }
