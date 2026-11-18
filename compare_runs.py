"""
====================================================================================================
compare_runs.py: SO SÁNH TRỰC QUAN HIỆU NĂNG VÀ CHẤT LƯỢNG GIỮA BASELINE VÀ LOW VRAM 3DGS
====================================================================================================
Cách sử dụng:
  py compare_runs.py results/truck_baseline results/truck_low_vram
  hoặc:
  py compare_runs.py --baseline results/truck_baseline --low_vram results/truck_low_vram --save_chart results/comparison.png
====================================================================================================
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Hỗ trợ hiển thị tiếng Việt trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_csv(path: str) -> List[Dict[str, str]]:
    """Đọc file CSV thành danh sách dict."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_summary(result_dir: str) -> Dict:
    """Đọc file benchmark_summary.json nếu có."""
    summary_path = os.path.join(result_dir, "benchmark_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def format_delta(base: float, target: float, unit: str = "", higher_is_better: bool = False) -> str:
    """Tính toán phần trăm và chênh lệch giữa hai giá trị."""
    if base == 0:
        return "N/A"
    diff = target - base
    pct = (diff / base) * 100.0

    sign = "+" if diff > 0 else ""
    # Xác định mũi tên trực quan
    better = (diff > 0) if higher_is_better else (diff < 0)
    tag = " (Tốt hơn)" if better and abs(pct) > 0.5 else ""

    return f"{sign}{pct:.1f}% ({sign}{diff:.2f}{unit}){tag}"


def print_comparison_table(base_dir: str, low_dir: str):
    """In bảng so sánh chi tiết giữa Baseline và Low VRAM."""
    base_summary = load_summary(base_dir)
    low_summary = load_summary(low_dir)

    base_train = load_csv(os.path.join(base_dir, "train_metrics.csv"))
    low_train = load_csv(os.path.join(low_dir, "train_metrics.csv"))

    base_eval = load_csv(os.path.join(base_dir, "eval_metrics.csv"))
    low_eval = load_csv(os.path.join(low_dir, "eval_metrics.csv"))

    # Lấy thông số đỉnh VRAM
    base_peak_vram = base_summary.get("peak_vram_mb")
    if base_peak_vram is None and base_train:
        base_peak_vram = max(float(r["vram_peak_mb"]) for r in base_train if "vram_peak_mb" in r)

    low_peak_vram = low_summary.get("peak_vram_mb")
    if low_peak_vram is None and low_train:
        low_peak_vram = max(float(r["vram_peak_mb"]) for r in low_train if "vram_peak_mb" in r)

    # Lấy VRAM tĩnh cuối cùng
    base_final_vram = base_summary.get("final_vram_allocated_mb")
    if base_final_vram is None and base_train:
        base_final_vram = float(base_train[-1]["vram_allocated_mb"])

    low_final_vram = low_summary.get("final_vram_allocated_mb")
    if low_final_vram is None and low_train:
        low_final_vram = float(low_train[-1]["vram_allocated_mb"])

    # Số lượng Gaussians cuối
    base_num_gs = base_summary.get("final_num_gs")
    if base_num_gs is None and base_train:
        base_num_gs = int(base_train[-1]["num_GS"])

    low_num_gs = low_summary.get("final_num_gs")
    if low_num_gs is None and low_train:
        low_num_gs = int(low_train[-1]["num_GS"])

    # Dung lượng checkpoint
    base_ckpt_size = base_summary.get("checkpoint_size_mb", 0.0)
    low_ckpt_size = low_summary.get("checkpoint_size_mb", 0.0)

    # Thời gian huấn luyện
    base_time_s = base_summary.get("total_time_seconds", 0.0)
    low_time_s = low_summary.get("total_time_seconds", 0.0)
    base_fps = base_summary.get("average_fps", 0.0)
    low_fps = low_summary.get("average_fps", 0.0)

    # Chất lượng cuối (PSNR, SSIM, LPIPS)
    base_psnr = base_summary.get("final_metrics", {}).get("psnr")
    if base_psnr is None and base_eval:
        base_psnr = float(base_eval[-1]["psnr"])

    low_psnr = low_summary.get("final_metrics", {}).get("psnr")
    if low_psnr is None and low_eval:
        low_psnr = float(low_eval[-1]["psnr"])

    base_ssim = base_summary.get("final_metrics", {}).get("ssim")
    if base_ssim is None and base_eval:
        base_ssim = float(base_eval[-1]["ssim"])

    low_ssim = low_summary.get("final_metrics", {}).get("ssim")
    if low_ssim is None and low_eval:
        low_ssim = float(low_eval[-1]["ssim"])

    base_lpips = base_summary.get("final_metrics", {}).get("lpips")
    if base_lpips is None and base_eval:
        base_lpips = float(base_eval[-1]["lpips"])

    low_lpips = low_summary.get("final_metrics", {}).get("lpips")
    if low_lpips is None and low_eval:
        low_lpips = float(low_eval[-1]["lpips"])

    rows = []

    # 1. VRAM
    if base_peak_vram and low_peak_vram:
        rows.append((
            "Đỉnh VRAM (Peak VRAM)",
            f"{base_peak_vram:.1f} MB",
            f"{low_peak_vram:.1f} MB",
            format_delta(base_peak_vram, low_peak_vram, " MB", higher_is_better=False),
        ))
    if base_final_vram and low_final_vram:
        rows.append((
            "VRAM tĩnh cuối (Allocated)",
            f"{base_final_vram:.1f} MB",
            f"{low_final_vram:.1f} MB",
            format_delta(base_final_vram, low_final_vram, " MB", higher_is_better=False),
        ))

    # 2. Checkpoint & Dung lượng
    if base_ckpt_size > 0 and low_ckpt_size > 0:
        rows.append((
            "Dung lượng Checkpoint (.pt)",
            f"{base_ckpt_size:.2f} MB",
            f"{low_ckpt_size:.2f} MB",
            format_delta(base_ckpt_size, low_ckpt_size, " MB", higher_is_better=False),
        ))

    # 3. Số lượng Gaussians
    if base_num_gs and low_num_gs:
        rows.append((
            "Số lượng Gaussians (N)",
            f"{base_num_gs:,}",
            f"{low_num_gs:,}",
            format_delta(base_num_gs, low_num_gs, "", higher_is_better=False),
        ))

    # 4. Chất lượng tái tạo
    if base_psnr is not None and low_psnr is not None:
        diff_psnr = low_psnr - base_psnr
        rows.append((
            "Chất lượng PSNR (val)",
            f"{base_psnr:.2f} dB",
            f"{low_psnr:.2f} dB",
            f"{'+' if diff_psnr > 0 else ''}{diff_psnr:.2f} dB",
        ))
    if base_ssim is not None and low_ssim is not None:
        diff_ssim = low_ssim - base_ssim
        rows.append((
            "Chỉ số SSIM (val)",
            f"{base_ssim:.4f}",
            f"{low_ssim:.4f}",
            f"{'+' if diff_ssim > 0 else ''}{diff_ssim:.4f}",
        ))
    if base_lpips is not None and low_lpips is not None:
        diff_lpips = low_lpips - base_lpips
        rows.append((
            "Chỉ số LPIPS (val)",
            f"{base_lpips:.4f}",
            f"{low_lpips:.4f}",
            f"{'+' if diff_lpips > 0 else ''}{diff_lpips:.4f}",
        ))

    # 5. Tốc độ & Thời gian
    if base_time_s > 0 and low_time_s > 0:
        rows.append((
            "Tổng thời gian huấn luyện",
            f"{base_time_s:.1f}s",
            f"{low_time_s:.1f}s",
            format_delta(base_time_s, low_time_s, "s", higher_is_better=False),
        ))
    if base_fps > 0 and low_fps > 0:
        rows.append((
            "Tốc độ trung bình (FPS)",
            f"{base_fps:.1f} step/s",
            f"{low_fps:.1f} step/s",
            format_delta(base_fps, low_fps, " fps", higher_is_better=True),
        ))

    # In ra terminal
    print("\n" + "=" * 92)
    print(f"{'BẢNG SO SÁNH ĐỐI ĐẦU: 3DGS BASELINE vs LOW VRAM (NHÓM 1)':^92}")
    print("=" * 92)
    print(f"{'Chỉ số đánh giá':<28} | {'Baseline':<18} | {'Low VRAM':<18} | {'Chênh lệch (Delta)'}")
    print("-" * 28 + "-+-" + "-" * 18 + "-+-" + "-" * 18 + "-+-" + "-" * 20)

    for metric, base_val, low_val, delta in rows:
        print(f"{metric:<28} | {base_val:<18} | {low_val:<18} | {delta}")

    print("=" * 92 + "\n")


def plot_comparison_curves(base_dir: str, low_dir: str, output_image: str):
    """Vẽ biểu đồ đường so sánh VRAM, Loss, và PSNR/SSIM."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Compare] Matplotlib chưa được cài đặt. Bỏ qua vẽ biểu đồ.")
        return

    base_train = load_csv(os.path.join(base_dir, "train_metrics.csv"))
    low_train = load_csv(os.path.join(low_dir, "train_metrics.csv"))

    base_eval = load_csv(os.path.join(base_dir, "eval_metrics.csv"))
    low_eval = load_csv(os.path.join(low_dir, "eval_metrics.csv"))

    if not base_train and not low_train:
        print("[Compare] Không tìm thấy dữ liệu train_metrics.csv để vẽ biểu đồ.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: VRAM theo Step
    ax = axes[0]
    if base_train and "vram_peak_mb" in base_train[0]:
        x = [int(r["step"]) for r in base_train]
        y_peak = [float(r["vram_peak_mb"]) for r in base_train]
        y_alloc = [float(r["vram_allocated_mb"]) for r in base_train]
        ax.plot(x, y_peak, label="Baseline Peak VRAM", color="crimson", linestyle="--")
        ax.plot(x, y_alloc, label="Baseline Allocated", color="lightcoral", alpha=0.7)

    if low_train and "vram_peak_mb" in low_train[0]:
        x = [int(r["step"]) for r in low_train]
        y_peak = [float(r["vram_peak_mb"]) for r in low_train]
        y_alloc = [float(r["vram_allocated_mb"]) for r in low_train]
        ax.plot(x, y_peak, label="Low VRAM Peak", color="royalblue", linewidth=2)
        ax.plot(x, y_alloc, label="Low VRAM Allocated", color="deepskyblue", alpha=0.7)

    ax.set_title("So sánh mức tiêu thụ VRAM (MB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("VRAM (MB)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Subplot 2: Loss Convergence
    ax = axes[1]
    if base_train and "loss" in base_train[0]:
        x = [int(r["step"]) for r in base_train]
        y = [float(r["loss"]) for r in base_train]
        ax.plot(x, y, label="Baseline Loss", color="crimson", alpha=0.8)

    if low_train and "loss" in low_train[0]:
        x = [int(r["step"]) for r in low_train]
        y = [float(r["loss"]) for r in low_train]
        ax.plot(x, y, label="Low VRAM Loss", color="royalblue", alpha=0.8)

    ax.set_title("Hội tụ hàm mất mát (Loss)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Subplot 3: PSNR Quality over Steps
    ax = axes[2]
    has_eval = False
    if base_eval and "psnr" in base_eval[0]:
        x = [int(r["step"]) for r in base_eval]
        y = [float(r["psnr"]) for r in base_eval]
        ax.plot(x, y, label="Baseline PSNR", color="crimson", marker="o", linewidth=2)
        has_eval = True

    if low_eval and "psnr" in low_eval[0]:
        x = [int(r["step"]) for r in low_eval]
        y = [float(r["psnr"]) for r in low_eval]
        ax.plot(x, y, label="Low VRAM PSNR", color="royalblue", marker="s", linewidth=2)
        has_eval = True

    ax.set_title("Chất lượng hình ảnh (PSNR dB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Evaluation Steps")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle=":", alpha=0.6)
    if has_eval:
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_image)), exist_ok=True)
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"[Compare] Đã lưu biểu đồ so sánh: {output_image}")


def main():
    parser = argparse.ArgumentParser(description="So sánh hiệu năng & chất lượng giữa Baseline và Low VRAM 3DGS")
    parser.add_argument("baseline_dir", nargs="?", default="results/truck_baseline", help="Thư mục kết quả Baseline")
    parser.add_argument("low_vram_dir", nargs="?", default="results/truck_low_vram", help="Thư mục kết quả Low VRAM")
    parser.add_argument("--baseline", dest="baseline_dir_opt", default=None, help="Thư mục Baseline (tùy chọn)")
    parser.add_argument("--low_vram", dest="low_vram_dir_opt", default=None, help="Thư mục Low VRAM (tùy chọn)")
    parser.add_argument("--save_chart", default="results/comparison_curves.png", help="Đường dẫn lưu file ảnh biểu đồ")
    args = parser.parse_args()

    base_dir = args.baseline_dir_opt or args.baseline_dir
    low_dir = args.low_vram_dir_opt or args.low_vram_dir

    if not os.path.exists(base_dir):
        print(f"[Cảnh báo] Chưa tìm thấy thư mục Baseline: {base_dir}")
    if not os.path.exists(low_dir):
        print(f"[Cảnh báo] Chưa tìm thấy thư mục Low VRAM: {low_dir}")

    print_comparison_table(base_dir, low_dir)

    if os.path.exists(base_dir) or os.path.exists(low_dir):
        plot_comparison_curves(base_dir, low_dir, args.save_chart)


if __name__ == "__main__":
    main()
