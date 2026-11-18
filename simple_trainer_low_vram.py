"""
====================================================================================================
simple_trainer_low_vram.py: 3D GAUSSIAN SPLATTING TRAINER TỐI ƯU VRAM (NHÓM 1)
Tích hợp trọn vẹn 4 trụ cột kỹ thuật vào cấu trúc chuẩn của gsplat simple_trainer.py:
  1. Progressive Lazy SH Allocation (Khởi tạo bậc l=0, mở rộng động l=1 -> 2 -> 3)
  2. Packed Rasterization (packed=True) + Sparse Gaussian Adam (sparse_grad=True)
  3. Mid-Training Trainable SH Codebook (K-Means GPU K=1024, mở rộng Bậc 3 trực tiếp trên Codebook, xóa sổ shN + Adam trên từng hạt)
  4. Lightweight Checkpoint & PLY Export chuẩn tương thích mọi viewer
  5. Hệ thống ghi log chuẩn hóa (CSV, TensorBoard, JSON summary) đồng bộ với baseline
  6. Bước 1: Importance Pruning & VRAM Budget Cap (Khóa trần số lượng hạt N, cắt tỉa đón đầu 3 chốt)
====================================================================================================
"""

import os
import sys

# Khắc phục hiển thị tiếng Việt trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo virtualenv Scripts có trong PATH (để tìm ninja.exe)
scripts_dir = os.path.dirname(sys.executable)
if scripts_dir and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

# Thiết lập môi trường CUDA và MSVC tương thích Windows + RTX 3050 (Compute 8.6)
os.environ.setdefault("CUDA_HOME", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("MAX_JOBS", "4")
if os.name == "nt":
    os.environ.setdefault("VCToolsVersion", "14.44.35207")
    os.environ.setdefault("VSCMD_ARG_VCVARS_VER", "14.44.35207")

# Thêm đường dẫn examples vào sys.path để import datasets, utils, gsplat_viewer độc lập
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(CURRENT_DIR, "gsplat_repo", "examples")
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

import gc
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim._functional as F_opt
import tqdm
import tyro
import viser
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from gsplat.losses import (
    depth_l1_loss,
    l1_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
    total_variation_loss,
)
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from train_logger import TrainLogger

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization, RasterizeMode

try:
    from gsplat.scene import GaussianScene
    from gsplat.stage import Stage
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"{e.name} is not installed. The example trainers require the "
        "scene/stage helper packages, which ship with gsplat."
    ) from e
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


# =====================================================================
# BỘ TỐI ƯU HÓA THƯA SparseGaussianAdam
# =====================================================================
class SparseGaussianAdam(torch.optim.Optimizer):
    """
    Bộ tối ưu hóa Adam chuyên dụng cho 3D Gaussian Splatting:
    - Khi nhận gradient thưa (sparse COO tensor từ rasterization packed=True, sparse_grad=True):
      sử dụng F_opt.sparse_adam chỉ cập nhật các hạt nhìn thấy trên camera frustum.
    - Khi nhận gradient dày (dense tensor, ví dụ Pha 3 khi tra cứu Codebook):
      tự động chuyển qua F_opt.adam một cách liền mạch.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        fused: bool = True,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, fused=fused)
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


# Đăng ký vào namespace gsplat.optimizers
import gsplat.optimizers
gsplat.optimizers.SparseGaussianAdam = SparseGaussianAdam


# =====================================================================
# THUẬT TOÁN K-MEANS CHIA LÔ TRÊN GPU (Trụ cột 3)
# =====================================================================
@torch.no_grad()
def chunked_gpu_kmeans(
    data: torch.Tensor,
    num_clusters: int = 1024,
    num_iters: int = 12,
    chunk_size: int = 25_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gom cụm ma trận [N, D] thành từ điển centroids [K_eff, D] (float32) và chỉ số indices [N] (int16/long).
    Chia theo chunk_size để tránh tạo ma trận khoảng cách [N, K] gây tràn VRAM trên GPU 4GB.
    Nếu N < num_clusters, tự động giới hạn số cụm hiệu dụng K_eff = N.
    """
    if data.numel() == 0 or data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(f"Input data for chunked_gpu_kmeans cannot be empty, got shape {data.shape}")

    N, D = data.shape
    device = data.device
    eff_clusters = min(num_clusters, N)

    perm = torch.randperm(N, device=device)[:eff_clusters]
    centroids = data[perm].clone()
    indices = torch.empty(N, dtype=torch.long, device=device)

    for _ in range(num_iters):
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dist = torch.cdist(data[start:end], centroids, p=2)
            indices[start:end] = dist.argmin(dim=1)

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(eff_clusters, device=device)

        new_centroids.index_add_(0, indices, data)
        counts.index_add_(0, indices, torch.ones(N, device=device))

        valid = counts > 0
        centroids[valid] = new_centroids[valid] / counts[valid].unsqueeze(1)

    return centroids, indices


# =====================================================================
# BƯỚC 1: BỘ ĐIỀU KHIỂN CẮT TỈA ĐÓN ĐẦU & KHÓA TRẦN HẠT (BUDGET CAP)
# =====================================================================
class ImportanceBudgetController:
    """
    Theo dõi điểm đóng góp S_i của từng hạt Gaussian theo thời gian:
      S_i = sum_t ( I(i in V_t) * sigma(o_i) * ln(1 + r_i) )
    Thực hiện:
      - Chốt 1: Khóa trần cứng số hạt (max_cap_points) trong quá trình Densification.
      - Chốt 2: Cắt tỉa đón đầu trước khi nâng bậc Lazy SH.
      - Chốt 3: Tinh lọc hạt rác trước khi gom cụm K-Means ở Pha 3.
    Đồng bộ hoàn chỉnh cả splats, optimizers và strategy_state qua gsplat.strategy.ops.remove.
    """

    def __init__(self, num_points: int, max_cap: Optional[int] = 500_000, device: str = "cuda", scene=None):
        self.device = device
        self.max_cap = max_cap
        self.scene = scene
        self.importance_score = torch.zeros(num_points, dtype=torch.float32, device=device)
        self.visibility_count = torch.zeros(num_points, dtype=torch.int32, device=device)

    @property
    def importance_score(self):
        return self.scene.signal["importance_score"] if self.scene is not None else self._importance_score

    @importance_score.setter
    def importance_score(self, value):
        if self.scene is not None:
            self.scene.signal["importance_score"] = value
        else:
            self._importance_score = value

    @property
    def visibility_count(self):
        return self.scene.signal["visibility_count"] if self.scene is not None else self._visibility_count

    @visibility_count.setter
    def visibility_count(self, value):
        if self.scene is not None:
            self.scene.signal["visibility_count"] = value
        else:
            self._visibility_count = value

    @torch.no_grad()
    def sync_size(self, current_points: int):
        """Mở rộng mảng điểm khi có thêm hạt mới sinh ra từ quá trình Densification."""
        old_N = self.importance_score.shape[0]
        if current_points > old_N:
            pad_len = current_points - old_N
            self.importance_score = torch.cat([
                self.importance_score,
                torch.zeros(pad_len, dtype=torch.float32, device=self.device),
            ])
            self.visibility_count = torch.cat([
                self.visibility_count,
                torch.zeros(pad_len, dtype=torch.int32, device=self.device),
            ])

    @torch.no_grad()
    def accumulate_step_stats(self, info: dict, opacities_logit: torch.Tensor):
        """Tích lũy điểm quan trọng S_i từ đầu ra rasterization."""
        if "gaussian_ids" not in info or info["gaussian_ids"] is None:
            return

        vis_ids = info["gaussian_ids"].reshape(-1)
        if vis_ids.numel() == 0:
            return

        cur_N = opacities_logit.shape[0]
        if cur_N > self.importance_score.shape[0]:
            self.sync_size(cur_N)

        valid = (vis_ids >= 0) & (vis_ids < self.importance_score.shape[0])
        vis_ids = vis_ids[valid]
        if vis_ids.numel() == 0:
            return

        radii = info["radii"].float()
        if radii.ndim > 1:
            radii = radii.amax(dim=-1)
        radii = radii.reshape(-1)[valid]
        vis_opacities = torch.sigmoid(opacities_logit[vis_ids].flatten())
        step_score = vis_opacities * torch.log1p(radii)

        self.importance_score.index_add_(0, vis_ids, step_score)
        self.visibility_count.index_add_(0, vis_ids, torch.ones_like(vis_ids, dtype=torch.int32))

    @torch.no_grad()
    def enforce_hard_cap(
        self,
        splats: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
        scene: Any,
        trainer: Optional[Any] = None,
    ) -> int:
        """
        CHỐT 1: Khóa trần ngân sách cứng (Hard Budget Cap).
        Khi N > max_cap, chọn lọc đúng top max_cap hạt có điểm đóng góp cao nhất,
        cắt tỉa chính xác (N - max_cap) hạt kém nhất.
        Tuyệt đối không bao giờ làm sụt giảm hạt ngoài ý muốn.
        """
        N = splats["means"].shape[0]
        if self.max_cap is None or N <= self.max_cap:
            return 0

        if self.importance_score.shape[0] != N:
            self.sync_size(N)

        opacities = torch.sigmoid(splats["opacities"].flatten())
        ranking_score = self.importance_score + opacities * 1e-3

        _, topk_idx = torch.topk(ranking_score, k=self.max_cap, largest=True)
        keep_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        keep_mask[topk_idx] = True

        is_prune = ~keep_mask
        n_prune = int(is_prune.sum().item())

        if n_prune > 0:
            remove(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                mask=is_prune,
                scene=scene,
            )
            if self.scene is None:
                self.importance_score = self.importance_score[keep_mask].contiguous()
                self.visibility_count = self.visibility_count[keep_mask].contiguous()

            if (
                trainer is not None
                and getattr(trainer, "is_compressed", False)
                and getattr(trainer, "sh_indices", None) is not None
            ):
                trainer.sh_indices = trainer.sh_indices[keep_mask].contiguous()

            torch.cuda.empty_cache()
            new_N = splats["means"].shape[0]
            print(f"[Chốt 1: Hard Budget Cap] Đã cắt tỉa {n_prune:,} hạt vượt trần. Số hạt N: {N:,} -> {new_N:,} (Khóa đúng trần {self.max_cap:,})")

        return n_prune

    @torch.no_grad()
    def prune_milestone(
        self,
        splats: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
        scene: Any,
        prune_quantile: float = 0.15,
        min_opacity: float = 0.005,
        max_prune_ratio: float = 0.25,
        trainer: Optional[Any] = None,
        reason: str = "Prune",
    ) -> int:
        """
        CHỐT 2 & 3: Cắt tỉa đón đầu trước các mốc nâng bậc SH hoặc nén Codebook.
        Loại bỏ các hạt có độ mờ quá thấp (< min_opacity) và phân vị điểm đóng góp thấp nhất.
        Có cơ chế chốt an toàn (Safety Guard): Không bao giờ cắt quá max_prune_ratio (25%) tổng số hạt.
        """
        N = splats["means"].shape[0]
        if self.importance_score.shape[0] != N:
            self.sync_size(N)

        opacities = torch.sigmoid(splats["opacities"].flatten())

        # 1. Hạt có độ mờ quá thấp (mặc định 0.005 theo prune_opa của DefaultStrategy)
        is_dead = opacities < min_opacity

        # 2. Cắt theo phân vị điểm quan trọng trong nhóm hạt đã từng nhìn thấy
        seen_mask = (~is_dead) & (self.visibility_count > 0)
        if seen_mask.sum() > 100 and prune_quantile > 0.0:
            score_thresh = torch.quantile(self.importance_score[seen_mask], prune_quantile)
            is_unimportant = seen_mask & (self.importance_score < score_thresh)
        else:
            is_unimportant = torch.zeros(N, dtype=torch.bool, device=self.device)

        is_prune = is_dead | is_unimportant

        # Chốt an toàn (Safety Guard): Tuyệt đối không cắt quá max_prune_ratio (25%) tổng số hạt
        max_allowed_prune = int(N * max_prune_ratio)
        if is_prune.sum() > max_allowed_prune:
            ranking_score = self.importance_score + opacities * 1e-3
            masked_score = torch.where(is_prune, ranking_score, torch.full_like(ranking_score, 1e9))
            _, prune_idx = torch.topk(masked_score, k=max_allowed_prune, largest=False)
            safe_prune_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            safe_prune_mask[prune_idx] = True
            is_prune = safe_prune_mask

        # 3. Nếu sau khi cắt tỉa mà vẫn vượt max_cap, kết hợp giữ đúng top max_cap
        keep_mask = ~is_prune
        num_keep = int(keep_mask.sum().item())
        if self.max_cap is not None and num_keep > self.max_cap:
            ranking_score = torch.where(
                keep_mask,
                self.importance_score + opacities * 1e-3,
                torch.full_like(self.importance_score, -1e9),
            )
            _, topk_idx = torch.topk(ranking_score, k=self.max_cap, largest=True)
            cap_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
            cap_mask[topk_idx] = True
            keep_mask = cap_mask
            is_prune = ~keep_mask

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                mask=is_prune,
                scene=scene,
            )
            if self.scene is None:
                self.importance_score = self.importance_score[keep_mask].contiguous()
                self.visibility_count = self.visibility_count[keep_mask].contiguous()

            if (
                trainer is not None
                and getattr(trainer, "is_compressed", False)
                and getattr(trainer, "sh_indices", None) is not None
            ):
                trainer.sh_indices = trainer.sh_indices[keep_mask].contiguous()

            torch.cuda.empty_cache()
            new_N = splats["means"].shape[0]
            print(f"[{reason}] Đã cắt tỉa {n_prune:,} hạt. Số hạt N: {N:,} -> {new_N:,} (-{(n_prune / N) * 100:.1f}%)")

        return n_prune


# =====================================================================
# CẤU HÌNH HUẤN LUYỆN (CONFIG)
# =====================================================================
@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Dataset backend: "colmap" or "ncore"
    data_type: str = "colmap"
    # Path to the dataset
    data_dir: str = "data/tandt/truck"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/truck_low_vram"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training
    patch_size: Optional[int] = None
    # Global scale
    global_scale: float = 1.0
    # Normalize world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata
    load_exposure: bool = True
    backend: str = "cuda"

    # NCore options
    ncore_camera_ids: List[str] = field(default_factory=list)
    ncore_lidar_ids: List[str] = field(default_factory=list)
    ncore_seek_offset_sec: Optional[float] = None
    ncore_duration_sec: Optional[float] = None
    ncore_max_lidar_points: int = 500_000
    ncore_lidar_color_generic_data_name: str = "rgb"
    ncore_poses_component_group: str = "default"
    ncore_intrinsics_component_group: str = "default"
    ncore_masks_component_group: str = "default"

    port: int = 8080
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    cache_images: bool = False

    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = True
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_fp16: bool = False
    # Giãn chu kỳ nâng bậc SH để ghìm VRAM trong giai đoạn nhân bản hạt bùng nổ
    sh_degree_interval: int = 2500

    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 1e10

    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )

    # TRỤ CỘT 2: Packed Rasterization + Sparse Adam (mặc định BẬT để tối ưu VRAM)
    packed: bool = True
    sparse_grad: bool = True
    visible_adam: bool = False
    antialiased: bool = False

    # TRỤ CỘT 1: Cấp phát bậc thang trễ cho hàm Hài cầu (Lazy SH Allocation)
    lazy_sh: bool = True
    # TRỤ CỘT 3: Nén từ điển Codebook giữa chu kỳ (Trainable SH Codebook)
    use_codebook: bool = True
    codebook_size: int = 1024
    codebook_step: Optional[int] = None
    kmeans_iters: int = 12
    kmeans_chunk: int = 25_000
    log_vram_every: int = 500

    # BƯỚC 1: Kiểm soát trần số lượng hạt & Cắt tỉa đón đầu (Importance Budget Cap)
    enable_budget_controller: bool = True
    max_cap_points: Optional[int] = 500_000
    prune_quantile: float = 0.15
    min_opacity_prune: float = 0.005

    random_bkgd: bool = False

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    bilateral_grid_fused: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    ppisp_use_controller: bool = True
    ppisp_controller_distillation: bool = True
    ppisp_controller_activation_num_steps: int = 25_000
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    use_color_correction_metric: bool = False

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = False
    log_every: int = 50
    lpips_net: Literal["vgg", "alex"] = "alex"
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        if self.codebook_step is not None:
            self.codebook_step = int(self.codebook_step * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


# =====================================================================
# KHỞI TẠO MÔ HÌNH VỚI LAZY SH (Trụ cột 1)
# =====================================================================
def create_splats_with_optimizers_low_vram(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = True,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    lazy_sh: bool = True,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Khởi tạo Gaussian với kiến trúc tiết kiệm bộ nhớ:
    - Nếu lazy_sh=True: Ở Pha 1 hoàn toàn không cấp phát shN ([N, 45]), tiết kiệm 77.4% VRAM tĩnh ban đầu.
    - Sử dụng SparseGaussianAdam làm bộ tối ưu khi sparse_grad=True.
    """
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError(f"Unknown init type: {init_type}")

    # Phân chia dữ liệu đa tiến trình nếu có
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]

    N = points.shape[0]
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    quats = torch.rand((N, 4))
    quats = F.normalize(quats, dim=-1)
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means", torch.nn.Parameter(points.to(device)), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales.to(device)), scales_lr),
        ("quats", torch.nn.Parameter(quats.to(device)), quats_lr),
        ("opacities", torch.nn.Parameter(opacities.to(device)), opacities_lr),
        ("sh0", torch.nn.Parameter(rgb_to_sh(rgbs).unsqueeze(1).to(device)), sh0_lr),
    ]

    # TRỤ CỘT 1: Nếu KHÔNG bật lazy_sh, mới cấp phát shN ngay từ đầu
    if not lazy_sh and sh_degree > 0:
        alloc_shN = torch.zeros((N, (sh_degree + 1) ** 2 - 1, 3), device=device)
        params.append(("shN", torch.nn.Parameter(alloc_shN), shN_lr))

    if feature_dim is not None:
        features = torch.rand((N, feature_dim), device=device)
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs.to(device))
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({name: param for name, param, _ in params})

    BS = batch_size
    optimizers = {}
    for name, _, lr in params:
        if name in ["sh0", "shN", "opacities"]:
            # Dùng Fused Adam trực tiếp cho SH & Opacities để chạy CUDA C++ kernel in-place, không tốn VRAM phụ
            optimizers[name] = torch.optim.Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                fused=True,
            )
        else:
            # means, scales, quats tương thích với sparse_grad của gsplat
            optimizer_class = SparseGaussianAdam if sparse_grad else torch.optim.Adam
            optimizers[name] = optimizer_class(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
    return splats, optimizers


# =====================================================================
# RUNNER CHÍNH CỦA BỘ HUẤN LUYỆN
# =====================================================================
class Runner:
    """Động cơ huấn luyện 3DGS tối ưu hóa Low VRAM (Nhóm 1)."""

    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: Config) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"{cfg.backend}:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Unified Train Logger
        self.logger = TrainLogger(
            result_dir=cfg.result_dir,
            exp_name=Path(cfg.result_dir).name,
            trainer_type="low_vram",
            log_every=cfg.log_every,
            is_rank_zero=(world_rank == 0),
            eval_only=(cfg.ckpt is not None),
        )

        # Tải dữ liệu Dataset
        if cfg.data_type == "ncore":
            if cfg.cache_images:
                raise ValueError("cache_images is not supported for NCore data.")
            from datasets.ncore import NCoreDataset, NCoreParser

            self.parser = NCoreParser(
                meta_json_path=cfg.data_dir,
                factor=1.0 / cfg.data_factor if cfg.data_factor > 1 else 1.0,
                test_every=cfg.test_every,
                camera_ids=cfg.ncore_camera_ids or None,
                lidar_ids=cfg.ncore_lidar_ids or None,
                seek_offset_sec=cfg.ncore_seek_offset_sec,
                duration_sec=cfg.ncore_duration_sec,
                max_lidar_points=cfg.ncore_max_lidar_points,
                lidar_color_generic_data_name=cfg.ncore_lidar_color_generic_data_name,
                poses_component_group=cfg.ncore_poses_component_group,
                intrinsics_component_group=cfg.ncore_intrinsics_component_group,
                masks_component_group=cfg.ncore_masks_component_group,
                normalize_world_space=cfg.normalize_world_space,
            )
            self.trainset = NCoreDataset(self.parser, split="train")
            self.valset = NCoreDataset(self.parser, split="val")
            self.ncore_camera_data = [
                self.parser.camera_render_data[cam_id]
                for cam_id in self.parser.camera_ids
            ]
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                load_exposure=cfg.load_exposure,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"[Trainer] Scene scale: {self.scene_scale}")

        # Khởi tạo mô hình Low VRAM
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers_low_vram(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            lazy_sh=cfg.lazy_sh,
        )

        self.scene = GaussianScene.from_splats(self.splats, id="scene")
        self.splats = self.scene.splats
        self.stage = Stage()
        self.stage.add_scene(self.scene, self.rasterize_splats)
        print(f"[Trainer] Model initialized. Number of GS: {len(self.splats['means']):,}")

        # Quản lý trạng thái Low VRAM (Trụ cột 1 & 3)
        self.active_sh_degree = 0 if cfg.lazy_sh else cfg.sh_degree
        self.is_compressed = False
        self.sh_codebook = None
        self.sh_indices = None
        self.codebook_optimizer = None

        # Bộ điều khiển ngân sách hạt & Cắt tỉa đón đầu (Bước 1)
        if cfg.enable_budget_controller:
            self.budget_controller = ImportanceBudgetController(
                num_points=len(self.splats["means"]),
                max_cap=cfg.max_cap_points,
                device=self.device,
                scene=self.scene,
            )
        else:
            self.budget_controller = None

        # Thiết lập bước nén Codebook (mặc định sau khi dừng nhân bản hạt)
        self.codebook_step = cfg.codebook_step
        if self.codebook_step is None:
            if isinstance(cfg.strategy, DefaultStrategy):
                self.codebook_step = cfg.strategy.refine_stop_iter
            else:
                self.codebook_step = 15_000

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Pose & App Optimizers
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.app_opt_reg,
                )
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        self.post_processing_optimizers = []
        self._gaussians_frozen = False

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, _ = self.rasterize_splats(
            camtoworlds=c2w.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=render_tab_state.sh_degree,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
        )

        render_colors = render_colors.squeeze(0)
        render_alphas = render_alphas.squeeze(0)

        if render_tab_state.render_mode == "rgb":
            colors = render_colors
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[..., 0]
            colors = apply_float_colormap(
                depth,
                colormap=render_tab_state.colormap,
                normalize=render_tab_state.normalize,
                min_val=render_tab_state.min_val,
                max_val=render_tab_state.max_val,
                inverted=render_tab_state.inverted,
            )
        elif render_tab_state.render_mode == "alpha":
            colors = apply_float_colormap(
                render_alphas,
                colormap=render_tab_state.colormap,
                normalize=render_tab_state.normalize,
                min_val=render_tab_state.min_val,
                max_val=render_tab_state.max_val,
                inverted=render_tab_state.inverted,
            )
        return colors.cpu().numpy()

    def upgrade_sh_degree(self, target_degree: int):
        """
        TRỤ CỘT 1: Mở rộng bậc thang SH trễ (Progressive Lazy Allocation).
        Chỉ cấp phát thêm tensor khi mô hình thực sự cần học chi tiết góc nhìn cao hơn.
        """
        assert target_degree > self.active_sh_degree
        N = len(self.splats["means"])
        prev_num_sh = (self.active_sh_degree + 1) ** 2 - 1
        new_num_sh = (target_degree + 1) ** 2 - 1
        diff = new_num_sh - prev_num_sh

        print(f"\n[Lazy SH] Nâng cấp SH degree từ l={self.active_sh_degree} -> l={target_degree} ({diff * 3} floats/hạt)...")

        new_feats = torch.zeros((N, diff, 3), device=self.device)

        if "shN" in self.splats:
            old_data = self.splats["shN"].data
            updated_data = torch.cat([old_data, new_feats], dim=1)
            self.splats["shN"] = torch.nn.Parameter(updated_data)
        else:
            self.splats["shN"] = torch.nn.Parameter(new_feats)

        BS = self.cfg.batch_size
        self.optimizers["shN"] = torch.optim.Adam(
            [{"params": self.splats["shN"], "lr": self.cfg.shN_lr * math.sqrt(BS), "name": "shN"}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )

        self.active_sh_degree = target_degree
        torch.cuda.empty_cache()

    def compress_sh_into_codebook(self, step: int):
        """
        TRỤ CỘT 3: Nén từ điển Codebook giữa chu kỳ (Mid-Training Codebook Compression).
        Gom cụm shN thành từ điển K=1024, sau đó mở rộng Codebook lên Bậc 3 (nếu cfg.sh_degree > active_sh_degree).
        Xóa bỏ hoàn toàn tensor shN và Adam states khổng lồ trên từng hạt.
        """
        # CHỐT 3: Tinh lọc hạt rác TRƯỚC KHI gom cụm K-Means (Bước 1)
        if self.budget_controller is not None:
            self.budget_controller.prune_milestone(
                self.splats,
                self.optimizers,
                self.strategy_state,
                self.scene,
                prune_quantile=self.cfg.prune_quantile,
                min_opacity=self.cfg.min_opacity_prune,
                trainer=self,
                reason="Chốt 3: Pre-Codebook Prune",
            )

        print(f"\n{'='*70}\n[Codebook] Bước {step}: Bắt đầu nén SH bằng GPU K-Means (K={self.cfg.codebook_size})...\n{'='*70}")
        N = len(self.splats["means"])

        # Lấy dữ liệu shN hiện tại
        if "shN" in self.splats and self.splats["shN"] is not None:
            shN_data = self.splats["shN"].detach()
            num_coeffs = shN_data.shape[1]
            flat_sh = shN_data.reshape(N, num_coeffs * 3)

            eff_clusters = min(self.cfg.codebook_size, N)
            centroids, indices = chunked_gpu_kmeans(
                flat_sh,
                num_clusters=self.cfg.codebook_size,
                num_iters=self.cfg.kmeans_iters,
                chunk_size=self.cfg.kmeans_chunk,
            )
            centroids = centroids.reshape(eff_clusters, num_coeffs, 3)
            if eff_clusters < self.cfg.codebook_size:
                pad = torch.zeros(
                    (self.cfg.codebook_size - eff_clusters, num_coeffs, 3),
                    device=centroids.device,
                    dtype=centroids.dtype,
                )
                centroids = torch.cat([centroids, pad], dim=0)
        else:
            num_coeffs = 0
            centroids = torch.zeros((self.cfg.codebook_size, 0, 3), device=self.device)
            indices = torch.zeros(N, dtype=torch.int16, device=self.device)

        # Mở rộng Codebook lên cfg.sh_degree (ví dụ mở rộng từ Bậc 2 lên Bậc 3 trực tiếp trên Codebook)
        target_degree = self.cfg.sh_degree
        target_num_coeffs = (target_degree + 1) ** 2 - 1
        if target_num_coeffs > num_coeffs:
            diff = target_num_coeffs - num_coeffs
            print(f"[Codebook] Mở rộng Codebook từ Bậc {self.active_sh_degree} ({num_coeffs} coeffs) lên Bậc {target_degree} ({target_num_coeffs} coeffs, thêm {diff} coeffs/cụm)...")
            extra_zeros = torch.zeros((self.cfg.codebook_size, diff, 3), device=centroids.device, dtype=centroids.dtype)
            centroids = torch.cat([centroids, extra_zeros], dim=1)
            self.active_sh_degree = target_degree
            mem_added_kb = self.cfg.codebook_size * diff * 3 * 4 / 1024.0
            print(f"[Codebook] Mở rộng thành công! Codebook shape: {list(centroids.shape)} (chỉ tốn thêm {mem_added_kb:.1f} KB VRAM)")

        self.sh_codebook = torch.nn.Parameter(centroids.to(self.device))
        self.sh_indices = indices.to(dtype=torch.int16, device=self.device)

        if "shN" in self.splats:
            del self.splats["shN"]
        if "shN" in self.optimizers:
            del self.optimizers["shN"]

        BS = self.cfg.batch_size
        self.codebook_optimizer = torch.optim.Adam(
            [{"params": self.sh_codebook, "lr": self.cfg.shN_lr * math.sqrt(BS), "name": "sh_codebook"}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )

        self.is_compressed = True
        torch.cuda.empty_cache()
        gc.collect()

        mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[Codebook] Hoàn tất nén & nâng cấp lên Bậc {self.active_sh_degree}! VRAM hiện tại: {mem_mb:.1f} MB\n{'='*70}\n")

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        splats: Optional[torch.nn.ParameterDict] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        splats = splats if splats is not None else self.splats
        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])

        sh_degree = kwargs.pop("sh_degree", self.active_sh_degree)
        image_ids = kwargs.pop("image_ids", None)

        if self.cfg.app_opt:
            colors = self.app_module(
                features=splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=sh_degree,
            )
            colors = colors + splats["colors"]
            colors = torch.sigmoid(colors)
            sh_degree_to_rasterize = None
        else:
            if self.is_compressed:
                reconstructed_shN = self.sh_codebook[self.sh_indices.long()]
                colors = torch.cat([splats["sh0"], reconstructed_shN], dim=1)
            elif self.active_sh_degree == 0:
                colors = splats["sh0"]
            else:
                colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

            if self.cfg.sh_fp16:
                colors = colors.half()

            # Đảm bảo sh_degree không vượt quá số hệ số màu thực tế trong tensor colors
            curr_max_degree = int(math.isqrt(colors.shape[1])) - 1
            sh_degree_to_rasterize = min(sh_degree, curr_max_degree)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        # Đảm bảo ma trận camera có dạng [..., 4, 4] tránh lỗi inv_ex trên 3x4
        if camtoworlds.shape[-2] == 3:
            bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=camtoworlds.device, dtype=camtoworlds.dtype)
            bottom = bottom.view(*([1] * (camtoworlds.ndim - 2)), 1, 4).expand(*camtoworlds.shape[:-2], 1, 4)
            camtoworlds = torch.cat([camtoworlds, bottom], dim=-2)

        sparse_grad = False if self.is_compressed else self.cfg.sparse_grad

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv_ex(camtoworlds).inverse,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy) and not self.is_compressed
                else False
            ),
            sparse_grad=sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            sh_degree=sh_degree_to_rasterize,
            **kwargs,
        )

        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps), desc="Huấn luyện Low VRAM", disable=world_rank != 0)

        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            exposure = data["exposure"].to(device) if "exposure" in data else None

            if cfg.pose_noise > 0.0:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # Lịch trình nâng cấp Lazy SH (Trụ cột 1)
            # Khi use_codebook=True: Giới hạn tối đa ở Bậc 2 trước khi nén để đè bẹp Peak VRAM trên toàn bộ hạt.
            # Bậc 3 sẽ được mở rộng trực tiếp trên 1,024 vector Codebook tại codebook_step!
            max_pre_degree = min(cfg.sh_degree, 2) if cfg.use_codebook else cfg.sh_degree
            sh_degree_to_use = min(step // cfg.sh_degree_interval, max_pre_degree)
            if cfg.lazy_sh and sh_degree_to_use > self.active_sh_degree and not self.is_compressed:
                # CHỐT 2: Cắt tỉa đón đầu TRƯỚC KHI cấp phát bộ nhớ cho bậc SH mới (Bước 1)
                if self.budget_controller is not None:
                    self.budget_controller.prune_milestone(
                        self.splats,
                        self.optimizers,
                        self.strategy_state,
                        self.scene,
                        prune_quantile=self.cfg.prune_quantile,
                        min_opacity=self.cfg.min_opacity_prune,
                        trainer=self,
                        reason=f"Chốt 2: Pre-SH(l={sh_degree_to_use}) Prune",
                    )
                self.upgrade_sh_degree(sh_degree_to_use)

            # Lịch trình nén Codebook (Trụ cột 3)
            if cfg.use_codebook and step == self.codebook_step and not self.is_compressed:
                self.compress_sh_into_codebook(step)

            # Forward pass
            renders, alphas, info = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.active_sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )

            # Tích lũy điểm quan trọng S_i từ đầu ra rasterization (Bước 1)
            if self.budget_controller is not None and not self.is_compressed:
                self.budget_controller.accumulate_step_stats(info, self.splats["opacities"].detach())

            colors = renders
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # Step pre-backward cho Densification
            if not self._gaussians_frozen and not self.is_compressed:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # Loss computation
            if masks is not None:
                l1loss = l1_loss(colors[masks], pixels[masks]).mean()
                colors_ssim = colors * masks[..., None]
                pixels_ssim = pixels * masks[..., None]
            else:
                l1loss = l1_loss(colors, pixels).mean()
                colors_ssim = colors
                pixels_ssim = pixels

            ssimloss = ssim_loss(
                colors_ssim.permute(0, 3, 1, 2),
                pixels_ssim.permute(0, 3, 1, 2),
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * opacity_reg_loss(self.splats["opacities"])
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * scale_reg_loss(self.splats["scales"])

            loss.backward()

            # Optimizer step
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Optimizer step cho Codebook ở Pha 3
            if self.is_compressed and self.codebook_optimizer is not None:
                self.codebook_optimizer.step()
                self.codebook_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                self.splats["quats"].data = F.normalize(self.splats["quats"].data, dim=-1)

            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Step post-backward (Densification - chỉ chạy trước khi nén Codebook)
            if not self._gaussians_frozen and not self.is_compressed:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        packed=cfg.packed,
                        scene=self.scene,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                        lr=schedulers[0].get_last_lr()[0],
                        scene=self.scene,
                    )

                # CHỐT 1: Khóa trần ngân sách hạt cứng (Budget Cap Enforcement - Bước 1)
                if self.budget_controller is not None:
                    cur_N = len(self.splats["means"])
                    self.budget_controller.sync_size(cur_N)
                    if self.cfg.max_cap_points is not None and cur_N > self.cfg.max_cap_points:
                        self.budget_controller.enforce_hard_cap(
                            self.splats,
                            self.optimizers,
                            self.strategy_state,
                            self.scene,
                            trainer=self,
                        )

            # Xác định giai đoạn huấn luyện (Pha 1, 2, 3)
            stage_str = (
                f"Pha 1: Lazy SH (l={self.active_sh_degree})"
                if not self.is_compressed and self.active_sh_degree == 0
                else (
                    f"Pha 2: Lazy SH (l={self.active_sh_degree})"
                    if not self.is_compressed
                    else f"Pha 3: Codebook l={self.active_sh_degree} (K={cfg.codebook_size})"
                )
            )

            # Ghi log huấn luyện chuẩn hóa ra CSV & TensorBoard
            if world_rank == 0:
                self.logger.log_train_step(
                    step=step,
                    loss=loss.item(),
                    l1_loss=l1loss.item(),
                    ssim_loss=ssimloss.item(),
                    num_GS=len(self.splats["means"]),
                    sh_degree=self.active_sh_degree,
                    phase=stage_str,
                    writer=self.writer,
                    force_log=(step == max_steps - 1),
                )

            # Log VRAM định kỳ lên tiến trình progress bar
            if step % cfg.log_vram_every == 0 or step == max_steps - 1:
                alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                pbar.set_postfix({
                    "GSs": f"{len(self.splats['means']):,}",
                    "VRAM": f"{alloc_mb:.1f}MB",
                    "Peak": f"{peak_mb:.1f}MB",
                    "Loss": f"{loss.item():.4f}",
                    "Phase": stage_str,
                })

            # Lưu Checkpoint nén (Trụ cột 4)
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                    "is_compressed": self.is_compressed,
                    "active_sh_degree": self.active_sh_degree,
                }
                print(f"\n[Checkpoint - Step {step}] {stats}")
                with open(f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w") as f:
                    json.dump(stats, f)

                data = {
                    "step": step,
                    "scene_id": self.scene.id,
                    "splats": self.splats.state_dict(),
                    "is_compressed": self.is_compressed,
                    "active_sh_degree": self.active_sh_degree,
                }
                if self.is_compressed:
                    data["sh_codebook"] = self.sh_codebook.detach().half().cpu()
                    data["sh_indices"] = self.sh_indices.cpu()

                ckpt_file = f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                torch.save(data, ckpt_file)
                file_size_mb = os.path.getsize(ckpt_file) / (1024 ** 2)
                print(f"[Checkpoint] Đã lưu mô hình (Dung lượng: {file_size_mb:.2f} MB)")

            # Lưu file PLY chuẩn tương thích các viewer
            if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                sh0 = self.splats["sh0"]

                if self.is_compressed:
                    shN = self.sh_codebook[self.sh_indices.long()]
                elif self.active_sh_degree == 0:
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    shN = self.splats["shN"]

                ply_path = f"{self.ply_dir}/point_cloud_{step}.ply"
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=ply_path,
                )
                print(f"[PLY Export] Đã xuất file {ply_path}")

            # Đánh giá full valset
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)

        if world_rank == 0:
            last_ckpt = f"{self.ckpt_dir}/ckpt_{max_steps - 1}_rank{self.world_rank}.pt"
            last_ply = f"{self.ply_dir}/point_cloud_{max_steps - 1}.ply"
            self.logger.finish(
                final_step=max_steps,
                final_num_gs=len(self.splats["means"]),
                ckpt_path=last_ckpt if os.path.exists(last_ckpt) else None,
                ply_path=last_ply if os.path.exists(last_ply) else None,
                config_dict={
                    "data_dir": cfg.data_dir,
                    "max_steps": max_steps,
                    "batch_size": cfg.batch_size,
                    "sh_degree": cfg.sh_degree,
                    "lazy_sh": cfg.lazy_sh,
                    "use_codebook": cfg.use_codebook,
                    "codebook_size": cfg.codebook_size,
                    "codebook_step": self.codebook_step,
                    "packed": cfg.packed,
                    "sparse_grad": cfg.sparse_grad,
                },
            )

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Đánh giá chất lượng hình ảnh (PSNR, SSIM, LPIPS)."""
        print("\n>>> Đang chạy đánh giá tập kiểm thử (Evaluation)...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.stage.render(
                self.scene.id,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.active_sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            })
            print(
                f"[Val Result] PSNR: {stats['psnr']:.3f} dB | "
                f"SSIM: {stats['ssim']:.4f} | "
                f"LPIPS: {stats['lpips']:.4f} | "
                f"Tốc độ: {stats['ellipse_time']:.3f}s/frame | "
                f"Số GS: {stats['num_GS']:,}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

            # Ghi log đánh giá chuẩn hóa ra CSV
            self.logger.log_eval_step(
                step=step,
                metrics=stats,
                num_GS=len(self.splats["means"]),
                eval_time_s=stats.get("ellipse_time", 0.0),
                writer=self.writer,
            )

    @torch.no_grad()
    def render_traj(self, step: int):
        """Xuất video quỹ đạo quan sát."""
        if self.cfg.disable_video:
            return
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 1)
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            radius = np.linalg.norm(camtoworlds_all[:, :2, 3], axis=-1).mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height, radius=radius)
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(camtoworlds_all)

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)

        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for camtoworld in tqdm.tqdm(camtoworlds_all, desc="Rendering Trajectory", disable=self.world_rank != 0):
            camtoworld = torch.from_numpy(camtoworld).float().to(device)
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=self.active_sh_degree,
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()

    def load_checkpoint(self, ckpt_paths: Union[str, List[str]]) -> int:
        """
        Nạp checkpoint (hỗ trợ cả uncompressed lẫn compressed codebook, đa shard rank).
        Tái thiết lập kích thước splats, active_sh_degree, codebook và tái liên kết scene/stage.
        """
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]

        ckpts = [
            torch.load(file, map_location=self.device, weights_only=False)
            for file in ckpt_paths
        ]

        first_splats = ckpts[0]["splats"]
        new_splats = {}
        for k in first_splats.keys():
            new_splats[k] = torch.nn.Parameter(
                torch.cat([ckpt["splats"][k] for ckpt in ckpts], dim=0).to(self.device)
            )
        self.splats = torch.nn.ParameterDict(new_splats)

        is_compressed = ckpts[0].get("is_compressed", False)
        self.is_compressed = is_compressed
        if is_compressed:
            self.active_sh_degree = ckpts[0].get("active_sh_degree", self.cfg.sh_degree)
            self.sh_codebook = torch.nn.Parameter(
                ckpts[0]["sh_codebook"].float().to(self.device)
            )
            self.sh_indices = torch.cat(
                [ckpt["sh_indices"] for ckpt in ckpts], dim=0
            ).to(self.device, dtype=torch.int16)
        else:
            self.sh_codebook = None
            self.sh_indices = None
            if "shN" in self.splats:
                num_coeffs = self.splats["shN"].shape[1]
                self.active_sh_degree = int(math.isqrt(num_coeffs + 1)) - 1
            else:
                self.active_sh_degree = 0

        # Tái tạo GaussianScene và Stage với các tensor kích thước mới
        scene_id = ckpts[0].get("scene_id", "scene")
        self.scene = GaussianScene.from_splats(self.splats, id=scene_id)
        self.splats = self.scene.splats
        self.stage = Stage()
        self.stage.add_scene(self.scene, self.rasterize_splats)

        if self.budget_controller is not None:
            self.budget_controller.scene = self.scene
            self.budget_controller.sync_size(len(self.splats["means"]))

        step = ckpts[0].get("step", 0)
        return step


# =====================================================================
# HÀM MAIN VÀ KHỞI ĐỘNG CLI
# =====================================================================
def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        step = runner.load_checkpoint(cfg.ckpt)
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper (Low VRAM optimized).",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo' (Low VRAM optimized).",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
