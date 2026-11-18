"""
====================================================================================================
PHƯƠNG ÁN KỸ THUẬT TỐI ƯU HÓA BỘ NHỚ VRAM (NHÓM 1) TRONG 3D GAUSSIAN SPLATTING
Triển khai 4 Trụ cột Kỹ thuật:
  1. Progressive Lazy SH Allocation (Cấp phát bậc thang trễ l=0 -> 1 -> 2 -> 3)
  2. Mid-Training Trainable SH Codebook (Nén K-Means GPU & xóa sổ shN + Adam giữa chu kỳ)
  3. Packed Rasterization + SparseGaussianAdam (Đóng gói thưa & giải phóng bộ nhớ đạo hàm)
  4. Selective Mixed-Precision & Lightweight Compressed Storage (~75% tiết kiệm dung lượng)
====================================================================================================
"""

import os
import sys

# Khắc phục lỗi hiển thị tiếng Việt trên console Windows (UnicodeEncodeError cp1252)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo đường dẫn Scripts của virtualenv (chứa ninja.exe) nằm trong PATH
scripts_dir = os.path.dirname(sys.executable)
if scripts_dir and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

# Cấu hình môi trường an toàn cho Windows + GPU kiến trúc Ampere (RTX 3050 - Compute 8.6)
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["MAX_JOBS"] = "4"
if os.name == "nt":
    os.environ.setdefault("VCToolsVersion", "14.44.35207")
    os.environ.setdefault("VSCMD_ARG_VCVARS_VER", "14.44.35207")

import gc
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim._functional as F_opt
from gsplat import rasterization
import gsplat.optimizers


# =====================================================================
# BỘ TỐI ƯU HÓA THƯA CHO HẠT GAUSSIAN (SparseGaussianAdam)
# Tương thích hoàn toàn cả Gradient thưa (COO sparse) và dày (Dense)
# =====================================================================
class SparseGaussianAdam(torch.optim.Optimizer):
    """
    Bộ tối ưu hóa Adam chuyên dụng cho 3D Gaussian Splatting:
    - Khi nhận gradient thưa (sparse COO tensor từ rasterization packed=True, sparse_grad=True):
      sử dụng F_opt.sparse_adam chỉ cập nhật các hạt nhìn thấy trên khung nhìn camera.
    - Khi nhận gradient dày (dense tensor, ví dụ Pha 3 khi tra cứu Codebook):
      tự động chuyển qua F_opt.adam mà không gây xung đột định dạng tensor.
    """
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-15):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Khởi tạo trạng thái động lượng bậc 1 (m) và bậc 2 (v) theo yêu cầu
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if grad.is_sparse:
                    F_opt.sparse_adam(
                        [p],
                        [grad],
                        [exp_avg],
                        [exp_avg_sq],
                        [step],
                        eps=eps,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        maximize=False,
                    )
                else:
                    step_t = torch.tensor(step, dtype=torch.float32, device=p.device)
                    F_opt.adam(
                        [p],
                        [grad],
                        [exp_avg],
                        [exp_avg_sq],
                        [],
                        [step_t],
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        weight_decay=0.0,
                        eps=eps,
                        maximize=False,
                    )
        return loss


# Đăng ký SparseGaussianAdam vào namespace gsplat.optimizers để đảm bảo tính đồng bộ
gsplat.optimizers.SparseGaussianAdam = SparseGaussianAdam


# =====================================================================
# 1. CẤU HÌNH SIÊU THAM SỐ TỐI ƯU VRAM NHÓM 1
# =====================================================================
@dataclass
class Group1Config:
    num_points: int = 300_000        # Số lượng hạt Gaussian thực nghiệm
    max_sh_degree: int = 3           # Bậc SH tối đa (bậc 3 -> 16 hàm, 48 floats/hạt)
    codebook_size: int = 1024        # Số lượng vector mẫu K trong từ điển SH (1024 cụm ~ 90KB)
    kmeans_iters: int = 12           # Số vòng lặp K-Means GPU
    kmeans_chunk: int = 25_000       # Chia lô K-Means để không tăng Peak VRAM trên GPU 4GB
    use_sparse_adam: bool = True     # Kích hoạt SparseGaussianAdam + sparse_grad
    use_packed_raster: bool = True   # Kích hoạt đóng gói hạt trong Frustum (packed=True)
    # Lịch trình chuyển giai đoạn (Progressive Lazy Allocation & Codebook Compression)
    step_unlock_sh1: int = 150       # Bắt đầu Pha 2a: Mở khóa SH bậc 1 (3 coeffs)
    step_unlock_sh2: int = 300       # Bắt đầu Pha 2b: Mở rộng SH bậc 2 (8 coeffs)
    step_unlock_sh3: int = 450       # Bắt đầu Pha 2c: Mở rộng SH bậc 3 (15 coeffs)
    step_compress_codebook: int = 600# Bắt đầu Pha 3: Gom cụm K-Means & xóa sổ shN + Adam
    total_steps: int = 800           # Tổng số bước thực nghiệm


# =====================================================================
# 2. THUẬT TOÁN K-MEANS CHIA LÔ TRÊN GPU (KHÔNG TỐN BỘ NHỚ PHỤ)
# =====================================================================
@torch.no_grad()
def chunked_gpu_kmeans(
    data: torch.Tensor,
    num_clusters: int = 1024,
    num_iters: int = 12,
    chunk_size: int = 25_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gom cụm ma trận [N, D] thành từ điển [K_eff, D] (float32) và mảng chỉ số [N] (int16).
    Chia theo chunk_size để tránh tạo ma trận khoảng cách [N, K] khổng lồ trên GPU.
    Nếu N < num_clusters, tự động giới hạn số cụm hiệu dụng K_eff = N.
    """
    if data.numel() == 0 or data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(f"Input data for chunked_gpu_kmeans cannot be empty, got shape {data.shape}")

    N, D = data.shape
    device = data.device
    eff_clusters = min(num_clusters, N)

    # Khởi tạo tâm cụm bằng cách lấy mẫu ngẫu nhiên từ các hạt hiện có
    perm = torch.randperm(N, device=device)[:eff_clusters]
    centroids = data[perm].clone()
    indices = torch.empty(N, dtype=torch.long, device=device)

    for _ in range(num_iters):
        # Gán cụm theo từng khúc (chunk) để không vượt trần VRAM
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dist = torch.cdist(data[start:end], centroids, p=2)
            indices[start:end] = dist.argmin(dim=1)

        # Cập nhật tâm cụm bằng phép cộng nguyên tử (atomic index_add) trên GPU
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(eff_clusters, device=device)

        new_centroids.index_add_(0, indices, data)
        counts.index_add_(0, indices, torch.ones(N, device=device))

        valid = counts > 0
        centroids[valid] = new_centroids[valid] / counts[valid].unsqueeze(1)

    return centroids, indices.to(torch.int16)


# =====================================================================
# 3. BỘ QUẢN LÝ HẠT GAUSSIAN TIẾT KIỆM VRAM (GROUP 1 GAUSSIAN MODEL)
# =====================================================================
class Group1GaussianModel:
    def __init__(self, cfg: Group1Config, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.N = cfg.num_points
        self.active_sh_degree = 0
        self.is_compressed = False
        self.sh_indices: Optional[torch.Tensor] = None

        # Khởi tạo Pha 1: TUYỆT ĐỐI KHÔNG CẤP PHÁT shN!
        # Chỉ lưu tâm (means), tỷ lệ (scales), xoay (quats), độ mờ (opacities) và màu gốc (sh0).
        torch.manual_seed(42)
        self.splats = nn.ParameterDict({
            "means": nn.Parameter((torch.rand(self.N, 3, device=device) - 0.5) * 2.5),
            "scales": nn.Parameter(torch.full((self.N, 3), math.log(0.02), device=device)),
            "quats": nn.Parameter(F.normalize(torch.randn(self.N, 4, device=device), dim=-1)),
            "opacities": nn.Parameter(torch.logit(torch.full((self.N,), 0.15, device=device))),
            "sh0": nn.Parameter(torch.rand(self.N, 1, 3, device=device)),
        })

        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._build_geometry_optimizers()

    def _create_opt(self, param: nn.Parameter, lr: float, allow_sparse: bool = True) -> torch.optim.Optimizer:
        """Tạo SparseGaussianAdam cho các thuộc tính per-Gaussian, hoặc Adam thường cho Codebook."""
        if self.cfg.use_sparse_adam and allow_sparse:
            return SparseGaussianAdam([param], lr=lr, eps=1e-15)
        return torch.optim.Adam([param], lr=lr, eps=1e-15)

    def _build_geometry_optimizers(self):
        lrs = {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
        }
        for name, lr in lrs.items():
            self.optimizers[name] = self._create_opt(self.splats[name], lr=lr, allow_sparse=True)

    # --- TRỤ CỘT 1: MỞ KHÓA BẬC THANG SH (PROGRESSIVE LAZY SH ALLOCATION) ---
    def upgrade_sh_degree(self, new_degree: int):
        assert new_degree > self.active_sh_degree and not self.is_compressed
        target_coeffs = (new_degree + 1) ** 2 - 1  # l=1 -> 3; l=2 -> 8; l=3 -> 15

        with torch.no_grad():
            if "shN" not in self.splats:
                new_shN = torch.zeros(self.N, target_coeffs, 3, device=self.device)
            else:
                old_shN = self.splats["shN"].detach()
                old_coeffs = old_shN.shape[1]
                pad_coeffs = target_coeffs - old_coeffs
                padding = torch.zeros(self.N, pad_coeffs, 3, device=self.device)
                new_shN = torch.cat([old_shN, padding], dim=1)
                del self.optimizers["shN"]
                del self.splats["shN"]

        self.splats["shN"] = nn.Parameter(new_shN)
        self.optimizers["shN"] = self._create_opt(
            self.splats["shN"], lr=2.5e-3 / 20.0, allow_sparse=True
        )
        self.active_sh_degree = new_degree

        gc.collect()
        torch.cuda.empty_cache()

    # --- TRỤ CỘT 2: NÉN CODEBOOK GIỮA QUÁ TRÌNH HUẤN LUYỆN (MID-TRAINING TRAINABLE SH CODEBOOK) ---
    def compress_sh_into_codebook(self):
        assert "shN" in self.splats and not self.is_compressed
        K = self.cfg.codebook_size
        num_rest = self.splats["shN"].shape[1]

        with torch.no_grad():
            flat_shN = self.splats["shN"].detach().reshape(self.N, -1)
            eff_K = min(K, self.N)
            centroids, indices = chunked_gpu_kmeans(
                flat_shN,
                num_clusters=K,
                num_iters=self.cfg.kmeans_iters,
                chunk_size=self.cfg.kmeans_chunk,
            )
            codebook_param = centroids.reshape(eff_K, num_rest, 3)
            if eff_K < K:
                pad = torch.zeros(
                    (K - eff_K, num_rest, 3),
                    device=centroids.device,
                    dtype=centroids.dtype,
                )
                codebook_param = torch.cat([codebook_param, pad], dim=0)

        # 1. Giải phóng hoàn toàn shN gốc và trạng thái Optimizer của nó (~76% VRAM tĩnh)
        del self.optimizers["shN"]
        del self.splats["shN"]

        # 2. Đăng ký bảng từ điển (Dùng Adam thường vì kích thước Kx15x3 chỉ ~180KB)
        self.splats["sh_codebook"] = nn.Parameter(codebook_param)
        self.sh_indices = indices  # Lưu int16 (2 bytes/hạt), không tốn bộ nhớ Autograd
        self.optimizers["sh_codebook"] = self._create_opt(
            self.splats["sh_codebook"], lr=1e-3, allow_sparse=False
        )
        self.is_compressed = True

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def compose_colors(self) -> torch.Tensor:
        """Tổ hợp màu dựa theo giai đoạn huấn luyện (Pha 1: sh0, Pha 2: cat, Pha 3: tra cứu khả vi)"""
        if self.active_sh_degree == 0:
            return self.splats["sh0"]
        if not self.is_compressed:
            return torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        # Pha 3: Giải nén khả vi thông qua chỉ số int16
        shN_from_book = self.splats["sh_codebook"][self.sh_indices.long()]
        return torch.cat([self.splats["sh0"], shN_from_book], dim=1)

    def step_all_optimizers(self):
        """Cập nhật trọng số và giải phóng hoàn toàn buffer đạo hàm (zero_grad set_to_none=True)"""
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Duy trì quaternions trên mặt cầu đơn vị (unit hypersphere) mà không làm ngắt autograd
        with torch.no_grad():
            self.splats["quats"].data = F.normalize(self.splats["quats"].data, dim=-1)

    def save_compressed_checkpoint(self, filepath: str):
        """Lưu checkpoint nén (chỉ lưu Codebook + indices int16, giảm ~75% kích thước file)"""
        checkpoint = {
            "means": self.splats["means"].detach().cpu(),
            "scales": self.splats["scales"].detach().cpu(),
            "quats": self.splats["quats"].detach().cpu(),
            "opacities": self.splats["opacities"].detach().cpu(),
            "sh0": self.splats["sh0"].detach().cpu(),
            "is_compressed": self.is_compressed,
            "active_sh_degree": self.active_sh_degree,
        }
        if self.is_compressed:
            checkpoint["sh_codebook"] = self.splats["sh_codebook"].detach().half().cpu()  # FP16
            checkpoint["sh_indices"] = self.sh_indices.cpu()  # int16
        elif "shN" in self.splats:
            checkpoint["shN"] = self.splats["shN"].detach().cpu()
        torch.save(checkpoint, filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
        print(f"[Export] Đã lưu mô hình nén vào '{filepath}' (Dung lượng: {file_size_mb:.2f} MB)")


# =====================================================================
# 4. HÀM ĐO ĐẠC CHỈ SỐ VRAM VÀ PSNR
# =====================================================================
def measure_vram_mb() -> Tuple[float, float]:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return alloc, peak


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return 100.0
    return -10.0 * math.log10(mse)


# =====================================================================
# 5. VÒNG LẶP HUẤN LUYỆN & XUẤT BẢNG SỐ LIỆU CHO BÁO CÁO ĐỒ ÁN
# =====================================================================
def main():
    cfg = Group1Config()
    device = "cuda"
    model = Group1GaussianModel(cfg, device=device)

    # Khởi tạo Camera & Ảnh đích mẫu 512x512
    H, W = 512, 512
    focal = 520.0
    K_mat = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], device=device).unsqueeze(0)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    viewmat[0, 2, 3] = 4.0

    y, x = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing="ij")
    gt_image = torch.stack([x, y, 0.5 * (x + y)], dim=-1).unsqueeze(0)

    print("=" * 102)
    print(f"THỰC NGHIỆM NHÓM 1: PROGRESSIVE LAZY SH + SPARSE ADAM + CODEBOOK ({cfg.num_points:,} HẠT GAUSSIAN)")
    print("=" * 102)
    print(f"{'Step':<6} | {'Giai đoạn hoạt động':<32} | {'VRAM Tĩnh (MB)':<16} | {'Peak VRAM (MB)':<16} | {'PSNR (dB)':<11} | {'Loss L1':<8}")
    print("-" * 102)

    start_time = time.perf_counter()

    for step in range(1, cfg.total_steps + 1):
        # Lịch trình chuyển giai đoạn tự động
        if step == cfg.step_unlock_sh1:
            model.upgrade_sh_degree(1)
        elif step == cfg.step_unlock_sh2:
            model.upgrade_sh_degree(2)
        elif step == cfg.step_unlock_sh3:
            model.upgrade_sh_degree(3)
        elif step == cfg.step_compress_codebook:
            model.compress_sh_into_codebook()

        colors = model.compose_colors()

        # Chỉ bật sparse_grad khi chưa nén thành Codebook
        # (Khi nén Codebook, PyTorch Autograd yêu cầu tensor dày để gom scatter-add vào sh_codebook)
        use_sparse_now = cfg.use_sparse_adam and (not model.is_compressed)

        # Lưu ý kỹ thuật: Truyền trực tiếp model.splats["quats"] vì gsplat đã chuẩn hóa nội bộ
        # Điều này tránh lỗi 'Sparse division requires a scalar divisor' của PyTorch khi sparse_grad=True
        render_colors, _, _ = rasterization(
            means=model.splats["means"],
            quats=model.splats["quats"],
            scales=torch.exp(model.splats["scales"]),
            opacities=torch.sigmoid(model.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K_mat,
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

        # In báo cáo tại các mốc quan trọng (đầu/cuối mỗi giai đoạn)
        if step in [1, 149, 150, 299, 300, 449, 450, 599, 600, 700, 800]:
            alloc_mb, peak_mb = measure_vram_mb()
            psnr = calc_psnr(render_colors, gt_image)
            if not model.is_compressed:
                stage_str = f"Pha {1 if model.active_sh_degree == 0 else 2}: Lazy SH (Bậc {model.active_sh_degree})"
            else:
                stage_str = f"Pha 3: Codebook (K={cfg.codebook_size})"
            print(f"{step:04d}   | {stage_str:<32} | {alloc_mb:13.2f} MB | {peak_mb:13.2f} MB | {psnr:8.2f} dB | {loss.item():.5f}")

    total_time = time.perf_counter() - start_time
    print("=" * 102)
    print(f"[Hoàn thành] Tổng thời gian thực nghiệm: {total_time:.2f} giây ({total_time / cfg.total_steps * 1000:.1f} ms/bước)")

    # Lưu checkpoint nén minh họa
    ckpt_path = "group1_compressed_model.pt"
    model.save_compressed_checkpoint(ckpt_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # Dọn dẹp file tạm sau khi kiểm thử xong


if __name__ == "__main__":
    main()
