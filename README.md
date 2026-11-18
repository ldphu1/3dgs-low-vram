# 3DGS Low-VRAM: Training 3D Gaussian Splatting on 4GB Consumer GPUs

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA12.4-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.4-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Hardware](https://img.shields.io/badge/Tested%20On-RTX%203050%20Laptop%20(4GB)-76b900.svg)](https://www.nvidia.com/en-us/geforce/gaming-laptops/)
[![gsplat](https://img.shields.io/badge/Rasterizer-gsplat%20v1.4.0+-brightgreen.svg)](https://github.com/nerfstudio-project/gsplat)
[![Tests](https://img.shields.io/badge/Unit%20Tests-14%2F14%20Passing-success.svg)](./tests/test_training_regressions.py)

An optimized, highly memory-efficient training framework for **3D Gaussian Splatting (3DGS)** designed to run full 30,000-step training directly on budget consumer GPUs with **as low as 4GB VRAM** (e.g. NVIDIA GeForce RTX 3050 Laptop).

---

## 🚀 Key Highlights & Benchmark Comparison

Tested on the standard **Tanks & Temples (`truck`)** dataset with full 30,000 iterations:

| Metric | Standard 3DGS Baseline | 3DGS Low-VRAM (Ours) | Impact / Savings |
| :--- | :---: | :---: | :---: |
| **Peak VRAM** | 1,279.7 MB | **614.6 MB** | **-52.0% VRAM** |
| **Active Allocated VRAM** | 643.1 MB | **192.9 MB** | **-70.0% VRAM** |
| **Training Speed** | 23.2 steps/s | **37.0 steps/s** | **+59% FPS (1.6x faster)** |
| **Total Training Time** | 21m 34s | **13m 31s** | **~8 minutes saved** |
| **Checkpoint Size (.pt)** | 187.8 MB | **23.6 MB** | **-87.4% Disk Storage** |
| **Active Gaussians** | 834,262 | **424,475** | **Optimized topology** |

---

## 🧠 Architectural Pillars

```
                     ┌─────────────────────────────────────────────────────────────┐
                     │               3DGS Low-VRAM Training Pipeline               │
                     └─────────────────────────────────────────────────────────────┘
                                                    │
        ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
        ▼                                           ▼                                           ▼
 ┌─────────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────────┐
 │   Pillar 1: Lazy SH     │               │  Pillar 2: Codebook     │               │ Pillar 3: Budget Cap    │
 │ (Progressive Allocation)│               │ (Chunked GPU K-Means)   │               │ (Importance Controller) │
 └─────────────────────────┘               └─────────────────────────┘               └─────────────────────────┘
  • Phase 1: Degree 0 (DC)                  • Cluster higher-order SH                 • Track particle signals
  • Phase 2: Degree 1 SH                    • 1024 trainable centroids                • Hard cap on particle count
  • Zero static SH overhead                 • int16 indices (2 bytes)                 • Preemptive pruning
```

### 1. Progressive Lazy SH Allocation (Pillars 1 & 2)
Standard 3DGS allocates 16 spherical harmonics (SH) coefficients per RGB channel ($16 \times 3 = 48$ floats) for every Gaussian right from step 0. 
- **Phase 1 (Steps 0 - 3,000):** Only DC terms ($l=0$, 3 floats/splat) are allocated.
- **Phase 2 (Steps 3,000 - 7,000):** Upgrades to $l=1$ (9 floats/splat) as geometry stabilizes.
- **Phase 3 (Steps 7,000+):** Upgrades to full $l=3$ directly inside the codebook.

### 2. Trainable Mid-Training SH Codebook (Pillar 2)
Higher-degree view-dependent colors are redundant across neighboring Gaussians. At Step 7,000:
- Uses a **Chunked GPU K-Means** algorithm that processes distance matrices in sub-chunks (`kmeans_chunk=25,000`), avoiding out-of-memory spikes.
- Clusters higher-order SH into a compact **1024-centroid trainable codebook**.
- Replaces $N \times 15 \times 3$ floating-point parameters with **2-byte `int16` indices**, saving **~76% static SH memory**.
- The codebook centroids remain fully differentiable and continuously update via Adam.

### 3. Importance Budget Controller & Hard Cap (Pillar 3)
Tracks the cumulative visibility and opacity contribution score:
$$S_i = \sum_t \left( \mathbb{I}(i \in V_t) \cdot \sigma(o_i) \cdot \ln(1 + r_i) \right)$$
- **Hard Budget Cap:** Caps maximum Gaussians (e.g. 500,000) during densification.
- **Preemptive Pruning:** Trims uninformative Gaussians before Lazy SH upgrades and K-Means clustering.
- Integrates seamlessly with `GaussianScene.signal` hooks so topology operations (`remove`, `duplicate`, `split`, `on_relocate`, `on_permute`) preserve score alignment.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Windows 11 / Linux
- NVIDIA GPU with $\ge 4\text{GB}$ VRAM (Compute Capability 8.0+)
- Python 3.10 or 3.11
- CUDA Toolkit 12.1 or 12.4

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ldphu1/3dgs-low-vram.git
cd 3dgs-low-vram

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
# source .venv/bin/activate # On Linux

# Install core PyTorch matching your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r gsplat_repo/examples/requirements.txt
```

> [!NOTE]
> For Windows 11 users compiling `gsplat` from source with MSVC and CUDA 12.4, use our pre-patched fork with CUB compatibility fixes: [https://github.com/ldphu1/gsplat](https://github.com/ldphu1/gsplat).

---

## 💻 Usage

### 1. Train Low-VRAM Model
```bash
python simple_trainer_low_vram.py default \
    --data_dir data/tandt/truck \
    --result_dir results/truck_low_vram \
    --max_steps 30000 \
    --codebook_size 1024 \
    --max_cap_points 500000 \
    --disable_viewer
```

### 2. Evaluate Checkpoint
Evaluation uses isolated logging (`eval_only_metrics.csv` & `eval_only_summary.json`), leaving training logs untouched:
```bash
python simple_trainer_low_vram.py default \
    --data_dir data/tandt/truck \
    --result_dir results/truck_low_vram \
    --ckpt results/truck_low_vram/ckpts/ckpt_29999_rank0.pt
```

### 3. Export to Web Viewer (.splat & SuperSplat .ply)
Exports directly using raw log-scale and logit-opacity parameterization:
```bash
python convert.py \
    --ckpt results/truck_low_vram/ckpts/ckpt_29999_rank0.pt \
    --out_dir results/truck_low_vram \
    --formats splat ply_compressed ply
```
View the resulting `.splat` file in real-time in any web browser at [antimatter15.com/splat](https://antimatter15.com/splat/) or SuperSplat.

### 4. Compare Benchmark Runs
Generate side-by-side metric charts (Loss, FPS, VRAM, Peak VRAM, PSNR):
```bash
python compare_runs.py results/truck_baseline results/truck_low_vram --save_chart results/comparison.png
```

---

## 🧪 Regression Test Suite

All 14 unit tests in `tests/test_training_regressions.py` run on standard Python `unittest`:

```bash
python -m unittest discover -s tests -v
```

Tests verify:
- **Importance Alignment:** Signal hook synchronization across deletions, duplicates, and splits.
- **Checkpoint Evaluation:** Compressed codebook restoration, shard concatenation, and render equivalence.
- **Export Parameterization:** Prevention of scale/opacity double-exponentiation in `convert.py`.
- **Evaluation Logs:** Non-destructive logging in eval-only mode.
- **Appearance Optimization:** Safe post-activation RGB rendering with `sh_degree=None`.
- **K-Means Edge Cases:** Robust handling of empty inputs and small cluster counts ($N < K$).

---

## 📜 License

This project is licensed under the Apache 2.0 License. Built on top of [gsplat](https://github.com/nerfstudio-project/gsplat).
