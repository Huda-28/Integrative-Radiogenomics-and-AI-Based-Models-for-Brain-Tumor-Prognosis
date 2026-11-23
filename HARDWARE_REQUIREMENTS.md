# Hardware & Storage Requirements (Training vs. Local Inference)

**Project:** Integrative Radiogenomics and AI-Based Models for Brain Tumor Prognosis (`M0–M6` Pipeline & Radiogenomics Lab Web App)

This document defines the **minimum** and **recommended** hardware specifications—including exact **CPU, System RAM, GPU/VRAM, and Free Disk Space**—needed to retrain all models locally, run single-patient MRI inference, and host the interactive web application on your own machine.

---

## 1. Quick Comparison: Training vs. Inference vs. Web App

| Workload Mode | Minimum CPU | Minimum System RAM | Minimum GPU / VRAM | Minimum Free Disk Space | Recommended Hardware | Estimated Runtime |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **Mode A: Full End-to-End 3D Retraining**<br>*(Phase 3 `nnU-Net-3D` / `SFF-Former` + `M6` 3D Foundation LoRA + `M0–M6` Nested CV)* | 8-core x86_64 or ARM64<br>(Ryzen 7 / Intel i7) | **32 GB RAM** | **16 GB VRAM NVIDIA GPU**<br>(RTX 4060 Ti 16GB / A4000 / V100 / T4 with `fp16`/`bf16`) | **200 GB free NVMe SSD**<br>*(250–300 GB if keeping raw DICOMs)* | 12–16 core CPU · 64 GB RAM · **24 GB VRAM** (RTX 3090 / 4090 / L4) · 300 GB NVMe SSD | ~18–36 hrs (3D nnU-Net) + ~3–5 hrs (M6 LoRA 5×5 Nested CV) |
| **Mode B: Two-Stage Feature-Cached Retraining**<br>*(Frozen BraTS Masks & Pre-Extracted 5-Habitat / Foundation Tokens → Retrain `AutoComBat`, `High-EPP`, `M0–M5`, `M6 Mamba-GMU`)* | 4–8 core CPU<br>(Intel i5 / Ryzen 5 / Apple M1+) | **16 GB RAM** | **6–8 GB VRAM** (RTX 3050 / 3060)<br>*or CPU-only fallback* | **40 GB free SSD**<br>*(60 GB recommended)* | 8-core CPU · 32 GB RAM · 8–12 GB VRAM (RTX 3060 12GB / 4060) | ~5–15 min on 8 GB GPU (~20–35 min on CPU) for full 5×5 Nested CV |
| **Mode C: Local Single-Patient MRI Inference**<br>*(Raw 3D DICOM/NIfTI → Preprocessing → 5-Habitat Mask → PyRadiomics → `M6` + `SHAP`)* | 4-core CPU<br>(Intel i5 / Apple M1) | **16 GB RAM** | **6–8 GB VRAM** (RTX 2060 / 3060) or 16 GB Apple Silicon *(CPU fallback supported)* | **20 GB free SSD**<br>*(30 GB recommended)* | 8-core CPU · 16–32 GB RAM · 8+ GB VRAM GPU | ~25–60 sec/patient on GPU (~3–7 min/patient on CPU) |
| **Mode D: Local Web App + Tabular Inference**<br>*(React 19 + Vite + Express Portal & Doctor Testing Sandbox)* | 2-core CPU | **4 GB RAM** | **None (CPU only)** | **1.5 GB free disk space**<br>*(2 GB recommended)* | 4-core CPU · 8 GB RAM · Node.js 20/22 LTS | Instant (< 100 ms per patient evaluation) |

---

## 2. Detailed Free Disk Space (Storage) Breakdown

### A. Software Environment & Dependencies
| Component | Disk Footprint | Notes |
| :--- | :---: | :--- |
| Python Virtual Environment (`.venv`) — CPU PyTorch stack | **~3.5 – 4.5 GB** | Core scientific, MONAI, PyRadiomics, lifelines, scikit-survival, SHAP |
| Python Virtual Environment (`.venv`) — NVIDIA CUDA 12.x stack | **~9.0 – 12.0 GB** | Includes PyTorch CUDA binaries (`cu121`/`cu124`), cuDNN, Transformers, PEFT |
| Web App (`node_modules` + Vite production `dist/`) | **~0.5 – 0.7 GB** | Node.js packages for React 19, Tailwind CSS, Express, Lucide |
| OS Pagefile / Virtual Memory Swap Buffer | **~4.0 – 8.0 GB** | Prevents out-of-memory crashes during 3D NIfTI resampling |

### B. Pretrained Model Weights & Checkpoints
| Component | Disk Footprint | Needed In Mode |
| :--- | :---: | :--- |
| BraTS-2021 5-Compartment 3D Segmentation Weights (`nnU-Net-3D` / `SFF-Former` 5-fold) | **~1.5 – 3.0 GB** | Mode A, B (if segmenting), C |
| 3D Brain MRI Foundation Backbone (LoRA base + rank adapters) | **~1.5 – 4.0 GB** | Mode A, B (token extraction), C |
| Trained `M0–M6` Survival Checkpoints, `AutoComBat` & Isotonic/Conformal Calibrators | **~0.2 – 0.5 GB** | Mode A, B, C |

### C. Neuroimaging Cohorts & Intermediate Caches
| Component | Disk Footprint | Needed In Mode |
| :--- | :---: | :--- |
| **BraTS-2021** (2,040 segmentation pretraining NIfTI cases) | **~45 – 60 GB** | Mode A only *(skip if using pretrained BraTS weights)* |
| **UPenn-GBM** (630 multiparametric structural + perfusion/diffusion studies) | **~55 – 75 GB** | Mode A *(raw MRI retraining)* |
| **TCGA-GBM + CPTAC-GBM** (discovery + locked external validation cohorts) | **~25 – 40 GB** | Mode A *(raw MRI retraining)* |
| **LUMIERE Longitudinal** (91 serial post-treatment follow-up RANO studies) | **~10 – 15 GB** | Mode A *(raw MRI longitudinal retraining)* |
| **Preprocessed 1 mm³ SRI24 Volumes, 5-Habitat Masks & nnU-Net Scratch Plans** | **~35 – 55 GB** | Mode A *(working SSD space during 3D preprocessing)* |
| **Extracted Feature Tensors & Token Cache** (`.parquet` / `.npz` / `.pt` embeddings) | **~2 – 6 GB** | Mode B *(compact cached features for fast retraining)* |
| **Single-Patient Temporary Working Directory** (DICOM → NIfTI → SRI24 → Masks) | **~0.5 – 2.0 GB** | Mode C *(per active patient scan)* |

### Summary: Minimum Free Disk Space Required Before You Start
- **Mode A (Full End-to-End 3D Retraining):** **200 GB free** on an internal **NVMe SSD** *(250–300 GB recommended; avoid slow external HDDs because nnU-Net 3D patch loading is I/O bound)*.
- **Mode B (Two-Stage Feature-Cached Retraining):** **40 GB free** SSD *(covers CUDA Python environment + pretrained checkpoints + extracted feature cache + experiment outputs)*.
- **Mode C (Single-Patient Local Inference Workstation):** **20 GB free** SSD *(covers CUDA/CPU Python environment + model weights + patient scan scratch space)*.
- **Mode D (Local Web App Only):** **1.5 GB free** disk space *(covers `node_modules`, Vite build, and static assets)*.

---

## 3. Component-by-Component Engineering Notes

- **Why Mode A Requires >= 16 GB VRAM:** 3D MRI patches (`128 × 128 × 128 × 4–6 channels`) plus the 3D foundation backbone need `fp16`/`bf16` mixed precision, gradient checkpointing, `batch_size=1–2`, and LoRA (`r=8–16`) to fit comfortably inside 16 GB VRAM (24 GB VRAM recommended).
- **Why Mode B & `M6` Mamba-GMU Run on 6–8 GB VRAM or CPU:** Training operates on pre-extracted token sequences (`5–10` tokens, `embed_dim=128`), and `SelectiveStateSpaceBlock` runs in pure PyTorch with no custom CUDA kernel compilation required.
- **Why Mode C Inference Fits in 6–8 GB VRAM:** Uses MONAI/nnU-Net sliding-window inference (`sw_batch_size=1`, `fp16`, `torch.inference_mode()`) keeping peak VRAM below ~5.5 GB, with automatic CPU fallback.
- **System RAM Workers:** Cap `pyradiomics` multiprocessing to `num_workers <= 4` on 32 GB RAM systems.
- **OS Compatibility:** Linux (Ubuntu 22.04/24.04) supports everything natively; Windows 10/11 runs Modes B/C/D natively (use WSL2 for full Mode A `antspyx` registration); macOS Apple Silicon (16 GB+ Unified Memory) supports Modes B/C/D natively.

---

## 4. Pre-Flight Local Check Commands

```bash
df -h .                     # Check free disk space (Linux / macOS)
python --version            # Recommended: Python 3.10 or 3.11
nvidia-smi                  # Check NVIDIA GPU & VRAM
python -c "import torch, monai, radiomics, lifelines, peft, shap; print('CUDA:', torch.cuda.is_available(), 'PyTorch:', torch.__version__)"
node -v && npm -v           # Requires Node.js >= 20 LTS for the Web App
```
