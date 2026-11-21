"""Phases 8-10: External Validation, Harrell's C-Index, Calibration & Decision Curve Analysis."""
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


def decision_curve_net_benefit(
    event_12m: np.ndarray,
    predicted_prob_12m: np.ndarray,
    threshold_pt: float = 0.30,
) -> float:
    """Calculates clinical Net Benefit at risk threshold p_t for Decision Curve Analysis."""
    n = float(len(event_12m))
    if n == 0:
        return 0.0
    treated = predicted_prob_12m >= threshold_pt
    tp = float(np.sum(treated & (event_12m == 1)))
    fp = float(np.sum(treated & (event_12m == 0)))
    odds = threshold_pt / max(1e-6, 1.0 - threshold_pt)
    return (tp / n) - (fp / n) * odds
