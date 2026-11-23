"""Local FastAPI Server exposing trained M5/M6 PyTorch checkpoints and metrics at http://localhost:8000."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from infer import run_patient_inference

CHECKPOINT_DIR = Path("./checkpoints")
HARMONIZER_PATH = CHECKPOINT_DIR / "harmonizer.pkl"
METRICS_PATH = CHECKPOINT_DIR / "metrics_summary.json"
PROGRESS_PATH = CHECKPOINT_DIR / "training_progress.json"

app = FastAPI(
    title="Radiogenomics M0–M6 Local Inference API",
    version="1.0.0",
    description="Serves trained AutoComBat + M5/M6 PyTorch survival & virtual-biopsy checkpoints locally.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInferenceRequest(BaseModel):
    patient_id: str = "LOCAL-WEB-CASE"
    site_id: str = "CPTAC_EXT"
    model: str = Field(default="m6", pattern="^(m5|m6)$")
    age: float = 62.0
    kps: int = 80
    extent_of_resection: int = 0
    idh_status: int = 0
    mgmt_status: int = -1
    tert_status: int = 1
    delta_t1ce_pct: float = 0.0
    delta_rcbv_pct: float = 0.0
    radiomics_overrides: Optional[Dict[str, float]] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "m5_checkpoint_ready": (CHECKPOINT_DIR / "m5_best.pt").exists(),
        "m6_checkpoint_ready": (CHECKPOINT_DIR / "m6_best.pt").exists(),
        "harmonizer_ready": HARMONIZER_PATH.exists(),
        "metrics_summary_ready": METRICS_PATH.exists(),
    }


@app.get("/metrics")
def metrics() -> dict:
    metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None
    progress_data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else None
    return {
        "metrics_summary": metrics_data,
        "training_progress": progress_data,
        "m5_checkpoint_ready": (CHECKPOINT_DIR / "m5_best.pt").exists(),
        "m6_checkpoint_ready": (CHECKPOINT_DIR / "m6_best.pt").exists(),
        "harmonizer_ready": HARMONIZER_PATH.exists(),
    }


@app.post("/predict")
def predict(req: PatientInferenceRequest) -> dict:
    ckpt_path = CHECKPOINT_DIR / f"{req.model}_best.pt"
    if not ckpt_path.exists() or not HARMONIZER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Trained checkpoint not found at {ckpt_path}. "
                "Run 'python train.py --model both --resume' first."
            ),
        )
    payload = req.model_dump(exclude={"model", "radiomics_overrides"})
    if req.radiomics_overrides:
        payload.update(req.radiomics_overrides)
    return run_patient_inference(payload, ckpt_path, HARMONIZER_PATH)
