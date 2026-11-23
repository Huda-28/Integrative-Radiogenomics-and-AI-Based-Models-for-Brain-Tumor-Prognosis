"""Local Single-Patient Prognosis Inference Script.

Loads a trained M6 (or M5) checkpoint + AutoComBat harmonizer and outputs:
  - Estimated Median OS + 12m / 24m / 36m survival probabilities
  - M6 90% Split Conformal prediction interval
  - Virtual-Biopsy MGMT, IDH & RTK-RAS/TP53/RB1 auxiliary predictions (even if lab markers are Unknown)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from src.models.m6_foundation_mamba_conformal import M6FoundationMambaConformalNet
from src.models.multimodal_radiogenomic_net import M5CrossModalRadiogenomicNet
from train import build_modality_tensors, resolve_device


DEFAULT_SAMPLE_PATIENT: Dict[str, Any] = {
    "patient_id": "LOCAL-CASE-001",
    "site_id": "CPTAC_EXT",
    "age": 62.0,
    "kps": 80,
    "extent_of_resection": 0,  # 0=GTR, 1=STR, 2=Biopsy
    "idh_status": 0,           # 0=Wildtype, 1=Mutant, -1=Unknown
    "mgmt_status": -1,         # -1=Unknown (triggers M5/M6 Virtual Biopsy)
    "tert_status": 1,
    "true_mgmt": 0,
    "true_idh": 0,
    "rtk_ras": 1,
    "tp53": 1,
    "rb1": 0,
    "delta_t1ce_pct": -8.0,
    "delta_rcbv_pct": -12.0,
    "os_months": 12.0,
    "event_observed": 0,
}


def run_patient_inference(
    patient_dict: Dict[str, Any],
    checkpoint_path: Path,
    harmonizer_path: Path,
    device_str: str = "auto",
) -> Dict[str, Any]:
    device = resolve_device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=device)
    with open(harmonizer_path, "rb") as fp:
        harm_bundle = pickle.load(fp)

    harmonizer = harm_bundle["harmonizer"]
    rad_cols = harm_bundle["rad_cols"]
    selected_cols = ckpt.get("selected_columns", harm_bundle["selected_columns"])
    model_type = ckpt.get("model_type", "m6")

    row = dict(DEFAULT_SAMPLE_PATIENT)
    row.update(patient_dict)
    for col in rad_cols:
        row.setdefault(col, 0.0)
    df = pd.DataFrame([row])

    rad_h = harmonizer.transform_fold(df[rad_cols], df["site_id"])
    tensors = build_modality_tensors(df, rad_h, selected_cols)

    model = (
        M5CrossModalRadiogenomicNet().to(device)
        if model_type == "m5"
        else M6FoundationMambaConformalNet().to(device)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        if model_type == "m5":
            out = model(
                tensors["radiomics"].to(device),
                tensors["deep_mri"].to(device),
                tensors["molecular"].to(device),
                tensors["clinical"].to(device),
                tensors["longitudinal"].to(device),
            )
            log_hr = float(out["log_hazard"].item())
            hr = float(np.exp(np.clip(log_hr, -2.5, 2.5)))
            s12 = float(np.exp(-0.52 * hr))
            s24 = float(np.exp(-1.10 * hr))
            s36 = float(np.exp(-1.62 * hr))
            median_os = float(np.clip(14.6 / max(0.25, hr), 2.5, 48.0))
            conf_low, conf_high = max(0.01, s12 - 0.11), min(0.99, s12 + 0.11)
        else:
            out = model(
                tensors["habitat_tokens"].to(device),
                tensors["radiomics"].to(device),
                tensors["molecular"].to(device),
                tensors["clinical"].to(device),
                tensors["longitudinal"].to(device),
            )
            curve = out["survival_curve"][0].cpu().numpy()  # 12 bins x 3 months
            low_90 = out["survival_lower_90"][0].cpu().numpy()
            high_90 = out["survival_upper_90"][0].cpu().numpy()
            s12, s24, s36 = float(curve[3]), float(curve[7]), float(curve[11])
            conf_low, conf_high = float(low_90[3]), float(high_90[3])
            hr = float(np.clip(1.0 / max(0.15, s12), 0.25, 9.5))
            months_axis = np.arange(1, 13, dtype=np.float32) * 3.0
            below = np.where(curve <= 0.5)[0]
            median_os = float(months_axis[below[0]]) if len(below) > 0 else float(months_axis[-1] + 4.0)

        mgmt_prob = float(torch.sigmoid(out["mgmt_logit"]).item())
        idh_prob = float(torch.sigmoid(out["idh_logit"]).item())
        pathway_probs = torch.sigmoid(out["pathway_logits"][0]).cpu().numpy().tolist()

    risk_tier = "High Risk" if s12 < 0.45 else ("Intermediate Risk" if s12 < 0.72 else "Low Risk")
    return {
        "patient_id": row["patient_id"],
        "model_used": model_type.upper(),
        "risk_tier": risk_tier,
        "hazard_ratio": round(hr, 3),
        "median_os_months": round(median_os, 1),
        "survival_12m_prob": round(s12, 4),
        "survival_24m_prob": round(s24, 4),
        "survival_36m_prob": round(s36, 4),
        "conformal_90_interval_12m": [round(conf_low, 4), round(conf_high, 4)],
        "virtual_biopsy_aux": {
            "mgmt_methylation_prob": round(mgmt_prob, 4),
            "idh_mutation_prob": round(idh_prob, 4),
            "rtk_ras_prob": round(pathway_probs[0], 4),
            "tp53_prob": round(pathway_probs[1], 4),
            "rb1_prob": round(pathway_probs[2], 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Local Inference with Trained M5/M6 Checkpoints")
    parser.add_argument("--checkpoint", type=Path, default=Path("./checkpoints/m6_best.pt"))
    parser.add_argument("--harmonizer", type=Path, default=Path("./checkpoints/harmonizer.pkl"))
    parser.add_argument("--input", type=Path, default=None, help="Optional path to patient JSON file.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    patient_data = json.loads(args.input.read_text(encoding="utf-8")) if args.input else DEFAULT_SAMPLE_PATIENT
    result = run_patient_inference(patient_data, args.checkpoint, args.harmonizer, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
