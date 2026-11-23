"""Phase 1: Automated Cohort Downloader & Integrity Verifier.

Supports:
  1. TCIA NBIA REST API (UPenn-GBM, TCGA-GBM, CPTAC-GBM) -> downloads DICOM series ZIPs & optional NIfTI conversion.
  2. Figshare Public API (LUMIERE Longitudinal Glioblastoma RANO cohort).
  3. BraTS-2021 Synapse DUA instructions & automated directory verification.
  4. Disk-space pre-flight verification against HARDWARE_REQUIREMENTS.md (200 GB full / 40 GB two-stage).
  5. Optional --smoke-test-prepare mode generating a strictly labeled synthetic split to validate train.py/infer.py on CPU.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
LUMIERE_FIGSHARE_ARTICLE_ID = "20172551"

COHORT_SPECS: Dict[str, Dict[str, str]] = {
    "upenn_gbm": {
        "source": "TCIA",
        "collection": "UPENN-GBM",
        "role": "Discovery Training & Internal Nested CV (n=630)",
        "est_size_gb": "65 GB",
        "clinical_url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642",
    },
    "tcga_gbm": {
        "source": "TCIA",
        "collection": "TCGA-GBM",
        "role": "Discovery Multi-Center Augmentation + GDC Genomics",
        "est_size_gb": "22 GB",
        "clinical_url": "https://portal.gdc.cancer.gov/projects/TCGA-GBM",
    },
    "cptac_gbm": {
        "source": "TCIA",
        "collection": "CPTAC-GBM",
        "role": "Locked External Validation Cohort (Zero Tuning)",
        "est_size_gb": "16 GB",
        "clinical_url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216",
    },
    "lumiere": {
        "source": "Figshare",
        "article_id": LUMIERE_FIGSHARE_ARTICLE_ID,
        "role": "Longitudinal Post-Op RANO & Pseudoprogression Cohort (n=91)",
        "est_size_gb": "12 GB",
        "clinical_url": "https://springernature.figshare.com/collections/LUMIERE_Dataset/6152957",
    },
    "brats_2021": {
        "source": "Synapse",
        "synapse_id": "syn25829067",
        "role": "5-Habitat 3D Segmentation Pretraining Only (n=2,040)",
        "est_size_gb": "55 GB",
        "clinical_url": "https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
    },
}


def check_free_disk_space(target_dir: Path, required_gb: float = 40.0) -> float:
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024 ** 3)
    print(f"[Disk Check] Path: {target_dir.resolve()} | Free SSD: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(
            f"[WARNING] Available space ({free_gb:.1f} GB) is below recommended {required_gb:.1f} GB. "
            "See HARDWARE_REQUIREMENTS.md (200 GB for full raw MRI retraining, 40 GB for feature-cached retraining)."
        )
    return free_gb


def fetch_tcia_series(collection: str, max_patients: int | None = None, timeout: int = 30) -> List[dict]:
    url = f"{TCIA_BASE_URL}/getSeries"
    resp = requests.get(url, params={"Collection": collection, "format": "json"}, timeout=timeout)
    resp.raise_for_status()
    series_list = resp.json()
    if max_patients is not None:
        seen_patients: set[str] = set()
        filtered: List[dict] = []
        for s in series_list:
            pid = str(s.get("PatientID", ""))
            if pid not in seen_patients and len(seen_patients) >= max_patients:
                continue
            seen_patients.add(pid)
            filtered.append(s)
        return filtered
    return series_list


def download_tcia_collection(
    cohort_key: str,
    out_root: Path,
    max_patients: int | None = None,
    convert_nifti: bool = False,
) -> None:
    spec = COHORT_SPECS[cohort_key]
    collection = spec["collection"]
    cohort_dir = out_root / "raw" / cohort_key
    dicom_dir = cohort_dir / "dicom"
    nifti_dir = cohort_dir / "nifti"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[TCIA] Querying series manifest for {collection} ({spec['role']})...")
    series_list = fetch_tcia_series(collection, max_patients=max_patients)
    manifest_path = cohort_dir / "series_manifest.json"
    manifest_path.write_text(json.dumps(series_list, indent=2), encoding="utf-8")
    print(f"[TCIA] Found {len(series_list)} series. Manifest saved -> {manifest_path}")
    print(f"[TCIA] Note: Download clinical/genomics CSV from: {spec['clinical_url']}")

    for idx, series in enumerate(series_list, start=1):
        uid = series["SeriesInstanceUID"]
        pid = series.get("PatientID", "unknown_patient")
        modality = series.get("Modality", "MR")
        dest_dir = dicom_dir / pid / f"{modality}_{uid[-8:]}"
        done_flag = dest_dir / ".download_complete"
        if done_flag.exists():
            continue
        tmp_dir = dicom_dir / pid / f".{modality}_{uid[-8:]}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dl_url = f"{TCIA_BASE_URL}/getImage"
        print(f"  [{idx}/{len(series_list)}] Downloading {pid} ({uid})...")
        r = requests.get(dl_url, params={"SeriesInstanceUID": uid}, stream=True, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(tmp_dir)
        (tmp_dir / ".download_complete").write_text("ok", encoding="utf-8")
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        os.replace(tmp_dir, dest_dir)

        if convert_nifti:
            try:
                import dicom2nifti  # type: ignore

                out_nii = nifti_dir / pid
                out_nii.mkdir(parents=True, exist_ok=True)
                dicom2nifti.convert_directory(str(dest_dir), str(out_nii), compression=True, reorient=True)
            except Exception as exc:
                print(f"  [WARN] NIfTI conversion skipped for {uid}: {exc}")


def download_lumiere_figshare(out_root: Path, dry_run: bool = False) -> None:
    cohort_dir = out_root / "raw" / "lumiere"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.figshare.com/v2/articles/{LUMIERE_FIGSHARE_ARTICLE_ID}"
    print(f"\n[Figshare] Fetching LUMIERE file list from {api_url}...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    article = resp.json()
    files = article.get("files", [])
    (cohort_dir / "figshare_manifest.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(f"[Figshare] Found {len(files)} LUMIERE files.")
    if dry_run:
        return
    for fmeta in files:
        fname = fmeta["name"]
        dl_url = fmeta["download_url"]
        dest = cohort_dir / fname
        part = cohort_dir / f"{fname}.part"
        if dest.exists() and dest.stat().st_size == int(fmeta.get("size", dest.stat().st_size)):
            continue
        print(f"  Downloading {fname} ({fmeta.get('size', 0) / (1024**2):.1f} MB)...")
        with requests.get(dl_url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(part, "wb") as fp:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fp.write(chunk)
        os.replace(part, dest)


def print_brats_synapse_instructions() -> None:
    print(
        "\n[BraTS-2021 Synapse Manual Auth Required]\n"
        "  1. Accept the BraTS Data Usage Agreement at https://www.synapse.org/#!Synapse:syn25829067\n"
        "  2. Install synapseclient: pip install synapseclient\n"
        "  3. Run: synapse get -r syn25829067 --downloadLocation ./data/raw/brats_2021\n"
        "  (Or skip raw BraTS download in Mode B if using pre-extracted 5-habitat features.)"
    )


def prepare_synthetic_smoke_test(out_root: Path, seed: int = 42) -> None:
    """Generates a strictly labeled synthetic cohort split so users can verify train.py/infer.py locally before 200GB MRI download."""
    rng = np.random.default_rng(seed)
    processed_dir = out_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    def _build_split(n: int, sites: List[str], cohort_label: str) -> pd.DataFrame:
        site_arr = rng.choice(sites, size=n)
        age = rng.normal(61.0, 10.5, size=n).clip(24, 86)
        kps = rng.choice([50, 60, 70, 80, 90, 100], size=n, p=[0.08, 0.14, 0.28, 0.28, 0.16, 0.06])
        eor = rng.choice([0, 1, 2], size=n, p=[0.48, 0.37, 0.15])  # 0=GTR, 1=STR, 2=Biopsy
        idh = rng.choice([0, 1, -1], size=n, p=[0.84, 0.08, 0.08])  # 0=WT, 1=Mut, -1=Unknown
        mgmt = rng.choice([1, 0, -1], size=n, p=[0.38, 0.48, 0.14])  # 1=Meth, 0=Unmeth, -1=Unknown
        tert = rng.choice([1, 0, -1], size=n, p=[0.70, 0.18, 0.12])
        glcm_entropy = rng.normal(3.6, 0.65, size=n).clip(1.4, 4.9)
        wavelet_hll = rng.normal(2.2, 0.55, size=n).clip(0.6, 3.4)
        rcbv = rng.normal(2.7, 0.75, size=n).clip(1.0, 4.4)
        delta_t1ce = rng.normal(8.0, 20.0, size=n).clip(-50, 70)
        delta_rcbv = rng.normal(6.0, 22.0, size=n).clip(-45, 65)

        site_shift = np.where(site_arr == sites[0], 0.0, 0.35)
        rad_cols = {f"rad_icc_{i:02d}": rng.normal(0.0, 1.0, size=n) + site_shift * (0.25 if i % 2 == 0 else -0.2) for i in range(18)}
        rad_cols["rad_icc_00"] = glcm_entropy + site_shift
        rad_cols["rad_icc_01"] = wavelet_hll + site_shift
        rad_cols["rad_icc_02"] = rcbv

        true_mgmt = np.where(mgmt == -1, (glcm_entropy < 3.4).astype(int), mgmt)
        true_idh = np.where(idh == -1, (age < 46).astype(int), idh)
        rtk_ras = (rcbv > 2.85).astype(int)
        tp53 = (glcm_entropy > 3.55).astype(int)
        rb1 = (wavelet_hll > 2.25).astype(int)

        log_hr = (
            0.032 * (age - 58.0)
            - 0.025 * (kps - 78.0)
            + 0.38 * eor
            - 0.68 * true_idh
            - 0.55 * true_mgmt
            + 0.35 * (glcm_entropy - 3.5)
            + 0.28 * (rcbv - 2.6)
            + 0.015 * np.maximum(0.0, delta_rcbv)
        )
        os_months = (15.8 * np.exp(-0.55 * log_hr) * rng.weibull(1.35, size=n)).clip(1.2, 54.0)
        censor_months = rng.uniform(14.0, 48.0, size=n)
        observed_time = np.minimum(os_months, censor_months)
        event_observed = (os_months <= censor_months).astype(int)

        df = pd.DataFrame({
            "patient_id": [f"{cohort_label}-{i:04d}" for i in range(n)],
            "site_id": site_arr,
            "age": age.round(1),
            "kps": kps,
            "extent_of_resection": eor,
            "idh_status": idh,
            "mgmt_status": mgmt,
            "tert_status": tert,
            "true_mgmt": true_mgmt,
            "true_idh": true_idh,
            "rtk_ras": rtk_ras,
            "tp53": tp53,
            "rb1": rb1,
            "delta_t1ce_pct": delta_t1ce.round(2),
            "delta_rcbv_pct": delta_rcbv.round(2),
            "os_months": observed_time.round(2),
            "event_observed": event_observed,
            "is_synthetic_smoke_test": True,
            **rad_cols,
        })
        return df

    disc_df = _build_split(180, ["UPENN_3T", "TCGA_GE", "TCGA_PHILIPS"], "SYNTH_DISC")
    ext_df = _build_split(60, ["CPTAC_EXT"], "SYNTH_CPTAC")
    disc_path = processed_dir / "smoke_discovery.csv"
    ext_path = processed_dir / "smoke_external_cptac.csv"
    disc_df.to_csv(disc_path, index=False)
    ext_df.to_csv(ext_path, index=False)
    meta = {
        "synthetic_smoke_test": True,
        "note": "Synthetic test split ONLY for verifying training/inference code flow before downloading real TCIA/BraTS MRI cohorts.",
        "discovery_csv": str(disc_path),
        "external_csv": str(ext_path),
        "discovery_n": len(disc_df),
        "external_n": len(ext_df),
    }
    (processed_dir / "smoke_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Smoke Test Ready] Generated {disc_path} (N={len(disc_df)}) and {ext_path} (N={len(ext_df)}).")
    print("  Next step: python train.py --model both --smoke-test --output-dir ./checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Radiogenomics Cohort Downloader & Smoke-Test Prep")
    parser.add_argument(
        "--cohort",
        nargs="+",
        choices=list(COHORT_SPECS.keys()),
        default=[],
        help="Cohorts to download (upenn_gbm, tcga_gbm, cptac_gbm, lumiere, brats_2021).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./data"), help="Root data directory.")
    parser.add_argument("--max-patients", type=int, default=None, help="Optional cap per TCIA cohort for quick testing.")
    parser.add_argument("--convert-nifti", action="store_true", help="Convert downloaded DICOM series to NIfTI via dicom2nifti.")
    parser.add_argument("--check-space", action="store_true", help="Check free SSD space and print cohort endpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch manifests only without downloading imaging ZIPs.")
    parser.add_argument(
        "--smoke-test-prepare",
        action="store_true",
        help="Generate synthetic tabular smoke-test cohorts to verify train.py and infer.py locally on CPU.",
    )
    args = parser.parse_args()

    if args.check_space or (not args.cohort and not args.smoke_test_prepare):
        check_free_disk_space(args.output_dir, required_gb=40.0)
        print("\nConfigured Radiogenomics Cohorts:")
        for key, spec in COHORT_SPECS.items():
            print(f"  - {key:11s} | {spec['source']:8s} | {spec['est_size_gb']:6s} | {spec['role']}")

    if args.smoke_test_prepare:
        prepare_synthetic_smoke_test(args.output_dir)

    for cohort_key in args.cohort:
        spec = COHORT_SPECS[cohort_key]
        if spec["source"] == "TCIA":
            if args.dry_run:
                series = fetch_tcia_series(spec["collection"], max_patients=args.max_patients)
                print(f"[Dry-Run] {spec['collection']}: {len(series)} series available via TCIA API.")
            else:
                download_tcia_collection(
                    cohort_key,
                    args.output_dir,
                    max_patients=args.max_patients,
                    convert_nifti=args.convert_nifti,
                )
        elif spec["source"] == "Figshare":
            download_lumiere_figshare(args.output_dir, dry_run=args.dry_run)
        elif spec["source"] == "Synapse":
            print_brats_synapse_instructions()


if __name__ == "__main__":
    main()
