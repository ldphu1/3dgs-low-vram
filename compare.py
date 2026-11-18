import os
import sys

# Khắc phục lỗi hiển thị tiếng Việt trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình môi trường an toàn cho Windows + RTX 3050
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["MAX_JOBS"] = "4"
if os.name == "nt":
    os.environ.setdefault("VCToolsVersion", "14.44.35207")
    os.environ.setdefault("VSCMD_ARG_VCVARS_VER", "14.44.35207")

import gc
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization

from train_low_vram import (
    SparseGaussianAdam,
    Group1Config,
    Group1GaussianModel,
    measure_vram_mb,
    calc_psnr,
)


# =====================================================================
# 1. MÔ PHỎNG CẤU TRÚC GỐC CỦA examples/simple_trainer.py (BASELINE)
# =====================================================================
def run_baseline_simple_trainer(num_points: int, H: int, W: int, K: torch.Tensor, viewmat: torch.Tensor,
                                gt_image: torch.Tensor):
    print("\n" + "=" * 80)
    print("1. ĐANG CHẠY BASELINE: CẤU TRÚC MẶC ĐỊNH CỦA simple_trainer.py (Full SH từ đầu)")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"

    # Khởi tạo giống hệt hàm create_splats_with_optimizers() trong simple_trainer.py
    torch.manual_seed(42)
    splats = nn.ParameterDict({
        "means": nn.Parameter((torch.rand(num_points, 3, device=device) - 0.5) * 2.0),
        "scales": nn.Parameter(torch.full((num_points, 3), math.log(0.02), device=device)),
        "quats": nn.Parameter(F.normalize(torch.randn(num_points, 4, device=device), dim=-1)),
        "opacities": nn.Parameter(torch.logit(torch.full((num_points,), 0.1, device=device))),
        "sh0": nn.Parameter(torch.rand(num_points, 1, 3, device=device)),
        "shN": nn.Parameter(torch.zeros(num_points, 15, 3, device=device)),  # <--- Cấp phát ngay từ Step 0!
    })

    optimizers = {
        "means": torch.optim.Adam([splats["means"]], lr=1.6e-4),
        "scales": torch.optim.Adam([splats["scales"]], lr=5e-3),
        "quats": torch.optim.Adam([splats["quats"]], lr=1e-3),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=2.5e-3),
        "shN": torch.optim.Adam([splats["shN"]], lr=2.5e-3 / 20.0),
    }

    records = {}
    start_time = time.perf_counter()

    for step in range(1, 601):
        # Trong simple_trainer.py gốc: colors luôn nối full [sh0, shN]
        sh_degree_to_use = 0 if step <= 200 else 3
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, 16, 3]

        render_colors, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree_to_use,
            packed=False,  # Mặc định của simple_trainer.py là packed=False
        )

        loss = F.l1_loss(render_colors, gt_image)
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=False)  # Mặc định giữ nguyên buffer grad

        with torch.no_grad():
            splats["quats"].data = F.normalize(splats["quats"].data, dim=-1)

        if step in [200, 400, 600]:
            alloc_mb, peak_mb = measure_vram_mb()
            psnr = calc_psnr(render_colors, gt_image)
            records[f"Pha {step // 200}"] = (alloc_mb, peak_mb, psnr)
            print(
                f"[Baseline | Step {step:03d}] VRAM Đang dùng: {alloc_mb:6.2f} MB | Peak VRAM: {peak_mb:6.2f} MB | PSNR: {psnr:5.2f} dB")

    total_time = time.perf_counter() - start_time
    records["time_s"] = total_time
    return records


# =====================================================================
# 2. CHẠY BẢN TỐI ƯU NHÓM 1: PROGRESSIVE LAZY SH + SPARSE ADAM + CODEBOOK
# =====================================================================
def run_optimized_low_vram(num_points: int, H: int, W: int, K: torch.Tensor, viewmat: torch.Tensor,
                           gt_image: torch.Tensor):
    print("\n" + "=" * 80)
    print("2. ĐANG CHẠY BẢN TỐI ƯU (NHÓM 1): PROGRESSIVE LAZY SH + SPARSE ADAM + CODEBOOK")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"

    # Cấu hình đồng bộ 600 bước cho bài toán đối chứng
    cfg = Group1Config(
        num_points=num_points,
        max_sh_degree=3,
        codebook_size=512,
        kmeans_iters=10,
        kmeans_chunk=25_000,
        use_sparse_adam=True,
        use_packed_raster=True,
        step_unlock_sh1=201,   # Mở khóa bậc 1 tại step 201
        step_unlock_sh2=270,   # Mở rộng bậc 2 tại step 270
        step_unlock_sh3=340,   # Mở rộng bậc 3 tại step 340
        step_compress_codebook=401,  # Gom cụm Codebook tại step 401
        total_steps=600,
    )

    model = Group1GaussianModel(cfg, device=device)
    records = {}
    start_time = time.perf_counter()

    for step in range(1, cfg.total_steps + 1):
        if step == cfg.step_unlock_sh1:
            torch.cuda.reset_peak_memory_stats()
            model.upgrade_sh_degree(1)
        elif step == cfg.step_unlock_sh2:
            model.upgrade_sh_degree(2)
        elif step == cfg.step_unlock_sh3:
            model.upgrade_sh_degree(3)
        elif step == cfg.step_compress_codebook:
            model.compress_sh_into_codebook()

        colors = model.compose_colors()
        use_sparse_now = cfg.use_sparse_adam and (not model.is_compressed)

        render_colors, _, _ = rasterization(
            means=model.splats["means"],
            quats=model.splats["quats"],
            scales=torch.exp(model.splats["scales"]),
            opacities=torch.sigmoid(model.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=model.active_sh_degree,
            packed=cfg.use_packed_raster,
            sparse_grad=use_sparse_now,
            absgrad=True,
        )

        loss = F.l1_loss(render_colors, gt_image)
        loss.backward()
        model.step_all_optimizers()

        if step in [200, 400, 600]:
            alloc_mb, peak_mb = measure_vram_mb()
            psnr = calc_psnr(render_colors, gt_image)
            records[f"Pha {step // 200}"] = (alloc_mb, peak_mb, psnr)
            print(
                f"[LowVRAM  | Step {step:03d}] VRAM Đang dùng: {alloc_mb:6.2f} MB | Peak VRAM: {peak_mb:6.2f} MB | PSNR: {psnr:5.2f} dB")

    total_time = time.perf_counter() - start_time
    records["time_s"] = total_time
    return records


# =====================================================================
# HÀM MAIN: CHẠY ĐỐI CHỨNG & XUẤT BẢNG SO SÁNH
# =====================================================================
def main():
    device = "cuda"
    N = 250_000
    H, W = 512, 512
    focal = 500.0
    K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], device=device).unsqueeze(0)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    viewmat[0, 2, 3] = 4.0

    # Tạo một bức ảnh mục tiêu mẫu có cấu trúc gradient mượt
    y, x = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij")
    gt_image = torch.stack([x, y, 0.5 * (x + y)], dim=-1).unsqueeze(0)

    # Chạy lần lượt 2 phương pháp
    base_res = run_baseline_simple_trainer(N, H, W, K, viewmat, gt_image)
    opt_res = run_optimized_low_vram(N, H, W, K, viewmat, gt_image)

    # In bảng tổng kết đối chứng
    print("\n" + "=" * 88)
    print(f"BẢNG TỔNG KẾT ĐỐI CHỨNG TRÊN {N:,} HẠT GAUSSIAN (ĐỘ PHÂN GIẢI {W}x{H})")
    print("=" * 88)
    print(
        f"{'Giai đoạn / Chỉ số':<26} | {'simple_trainer Gốc':<20} | {'Bản Tối ưu (Nhóm 1)':<20} | {'Chênh lệch':<15}")
    print("-" * 88)

    for phase in ["Pha 1", "Pha 2", "Pha 3"]:
        b_alloc, b_peak, b_psnr = base_res[phase]
        o_alloc, o_peak, o_psnr = opt_res[phase]
        diff_alloc = ((o_alloc - b_alloc) / b_alloc) * 100.0
        diff_peak = ((o_peak - b_peak) / b_peak) * 100.0

        print(f"{phase + ' - VRAM Đang dùng':<26} | {b_alloc:15.2f} MB | {o_alloc:15.2f} MB | {diff_alloc:+13.1f} %")
        print(f"{phase + ' - Peak VRAM':<26} | {b_peak:15.2f} MB | {o_peak:15.2f} MB | {diff_peak:+13.1f} %")
        print(
            f"{phase + ' - Chất lượng PSNR':<26} | {b_psnr:15.2f} dB | {o_psnr:15.2f} dB | {o_psnr - b_psnr:+12.2f} dB")
        print("-" * 88)

    b_time = base_res["time_s"]
    o_time = opt_res["time_s"]
    diff_time = ((o_time - b_time) / b_time) * 100.0
    print(f"{'Tổng thời gian (600 steps)':<26} | {b_time:16.2f} s | {o_time:16.2f} s | {diff_time:+13.1f} %")
    print("=" * 88)


if __name__ == "__main__":
    main()
