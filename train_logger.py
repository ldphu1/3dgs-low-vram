"""
====================================================================================================
train_logger.py: BỘ GHI LOG HUẤN LUYỆN CHUẨN HÓA CHO 3DGS (BASELINE & LOW VRAM)
====================================================================================================
Cung cấp:
  1. Ghi log step-by-step ra file CSV (`train_metrics.csv`)
  2. Ghi log đánh giá ra file CSV (`eval_metrics.csv`)
  3. Đồng bộ hóa các thẻ tag chuẩn trên TensorBoard (VRAM, Loss, FPS, PSNR, SSIM, LPIPS)
  4. Xuất báo cáo tổng kết chi tiết (`benchmark_summary.json` & `benchmark_summary.txt`)
====================================================================================================
"""

import csv
import json
import os
import time
from typing import Any, Dict, Optional
import torch


class TrainLogger:
    """Bộ quản lý và ghi log chuẩn hóa cho quá trình huấn luyện 3D Gaussian Splatting."""

    def __init__(
        self,
        result_dir: str,
        exp_name: str = "3dgs_experiment",
        trainer_type: str = "baseline",  # 'baseline' hoặc 'low_vram'
        log_every: int = 50,
        is_rank_zero: bool = True,
        eval_only: bool = False,
    ):
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.trainer_type = trainer_type
        self.log_every = log_every
        self.is_rank_zero = is_rank_zero
        self.eval_only = eval_only

        os.makedirs(self.result_dir, exist_ok=True)

        if self.eval_only:
            self.train_csv_path = None
            self.eval_csv_path = os.path.join(self.result_dir, "eval_only_metrics.csv")
            self.summary_json_path = os.path.join(self.result_dir, "eval_only_summary.json")
            self.summary_txt_path = os.path.join(self.result_dir, "eval_only_summary.txt")
        else:
            self.train_csv_path = os.path.join(self.result_dir, "train_metrics.csv")
            self.eval_csv_path = os.path.join(self.result_dir, "eval_metrics.csv")
            self.summary_json_path = os.path.join(self.result_dir, "benchmark_summary.json")
            self.summary_txt_path = os.path.join(self.result_dir, "benchmark_summary.txt")

        self.start_time = time.time()
        self.last_step_time = time.time()
        self.step_durations = []
        self.peak_vram_overall_mb = 0.0

        self.train_csv_file = None
        self.train_csv_writer = None
        self.eval_csv_file = None
        self.eval_csv_writer = None

        self.last_eval_metrics: Dict[str, float] = {}

        if self.is_rank_zero:
            self._init_csv()

    def _init_csv(self):
        """Khởi tạo các file CSV và ghi tiêu đề cột."""
        if not self.eval_only:
            # Train CSV
            train_headers = [
                "step",
                "time_elapsed_s",
                "step_time_ms",
                "fps",
                "num_GS",
                "vram_allocated_mb",
                "vram_reserved_mb",
                "vram_peak_mb",
                "loss",
                "l1_loss",
                "ssim_loss",
                "sh_degree",
                "phase",
            ]
            self.train_csv_file = open(self.train_csv_path, mode="w", newline="", encoding="utf-8")
            self.train_csv_writer = csv.DictWriter(self.train_csv_file, fieldnames=train_headers)
            self.train_csv_writer.writeheader()
            self.train_csv_file.flush()

        # Eval CSV
        eval_headers = [
            "step",
            "time_elapsed_s",
            "num_GS",
            "psnr",
            "ssim",
            "lpips",
            "eval_time_s",
            "vram_peak_mb",
        ]
        self.eval_csv_file = open(self.eval_csv_path, mode="w", newline="", encoding="utf-8")
        self.eval_csv_writer = csv.DictWriter(self.eval_csv_file, fieldnames=eval_headers)
        self.eval_csv_writer.writeheader()
        self.eval_csv_file.flush()

    def log_train_step(
        self,
        step: int,
        loss: float,
        l1_loss: float,
        ssim_loss: float,
        num_GS: int,
        sh_degree: int = 3,
        phase: str = "Standard",
        writer: Optional[Any] = None,
        force_log: bool = False,
    ):
        """Ghi thông số bước huấn luyện ra CSV và TensorBoard."""
        now = time.time()
        step_duration = now - self.last_step_time
        self.last_step_time = now
        self.step_durations.append(step_duration)

        if not self.is_rank_zero or self.eval_only:
            return

        # Chỉ ghi định kỳ log_every hoặc khi force_log (bước cuối)
        if not force_log and (step % self.log_every != 0):
            return

        elapsed_s = now - self.start_time
        recent_durations = self.step_durations[-self.log_every:]
        avg_step_duration = sum(recent_durations) / max(len(recent_durations), 1)
        step_time_ms = avg_step_duration * 1000.0
        fps = 1.0 / max(avg_step_duration, 1e-6)

        # Đo VRAM PyTorch
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if vram_peak > self.peak_vram_overall_mb:
                self.peak_vram_overall_mb = vram_peak
        else:
            vram_alloc, vram_res, vram_peak = 0.0, 0.0, 0.0

        # Ghi CSV
        row = {
            "step": step,
            "time_elapsed_s": round(elapsed_s, 2),
            "step_time_ms": round(step_time_ms, 2),
            "fps": round(fps, 1),
            "num_GS": num_GS,
            "vram_allocated_mb": round(vram_alloc, 2),
            "vram_reserved_mb": round(vram_res, 2),
            "vram_peak_mb": round(vram_peak, 2),
            "loss": round(float(loss), 5),
            "l1_loss": round(float(l1_loss), 5),
            "ssim_loss": round(float(ssim_loss), 5),
            "sh_degree": sh_degree,
            "phase": phase,
        }
        self.train_csv_writer.writerow(row)
        self.train_csv_file.flush()

        # Đồng bộ TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("train/l1_loss", float(l1_loss), step)
            writer.add_scalar("train/ssim_loss", float(ssim_loss), step)
            writer.add_scalar("train/num_GS", num_GS, step)
            writer.add_scalar("vram/allocated_mb", vram_alloc, step)
            writer.add_scalar("vram/reserved_mb", vram_res, step)
            writer.add_scalar("vram/peak_mb", vram_peak, step)
            writer.add_scalar("perf/step_time_ms", step_time_ms, step)
            writer.add_scalar("perf/fps", fps, step)

    def log_eval_step(
        self,
        step: int,
        metrics: Dict[str, float],
        num_GS: int,
        eval_time_s: float = 0.0,
        writer: Optional[Any] = None,
    ):
        """Ghi kết quả đánh giá (PSNR, SSIM, LPIPS) ra CSV và TensorBoard."""
        if not self.is_rank_zero:
            return

        elapsed_s = time.time() - self.start_time
        vram_peak = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        if vram_peak > self.peak_vram_overall_mb:
            self.peak_vram_overall_mb = vram_peak

        psnr = float(metrics.get("psnr", 0.0))
        ssim = float(metrics.get("ssim", 0.0))
        lpips = float(metrics.get("lpips", 0.0))

        self.last_eval_metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
        }

        row = {
            "step": step,
            "time_elapsed_s": round(elapsed_s, 2),
            "num_GS": num_GS,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "lpips": round(lpips, 4),
            "eval_time_s": round(eval_time_s, 3),
            "vram_peak_mb": round(vram_peak, 2),
        }
        self.eval_csv_writer.writerow(row)
        self.eval_csv_file.flush()

        if writer is not None:
            writer.add_scalar("val/psnr", psnr, step)
            writer.add_scalar("val/ssim", ssim, step)
            writer.add_scalar("val/lpips", lpips, step)
            writer.add_scalar("val/eval_time_s", eval_time_s, step)

    def finish(
        self,
        final_step: int,
        final_num_gs: int,
        ckpt_path: Optional[str] = None,
        ply_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Hoàn tất quá trình huấn luyện, tạo báo cáo JSON & TXT tổng kết."""
        if not self.is_rank_zero:
            return {}

        total_time_s = time.time() - self.start_time
        hours = int(total_time_s // 3600)
        minutes = int((total_time_s % 3600) // 60)
        seconds = int(total_time_s % 60)
        formatted_time = f"{hours:02d}h {minutes:02d}m {seconds:02d}s" if hours > 0 else f"{minutes:02d}m {seconds:02d}s"

        avg_step_ms = (total_time_s / max(final_step, 1)) * 1000.0
        avg_fps = max(final_step, 1) / max(total_time_s, 1e-6)

        final_vram_alloc = (
            torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        )
        overall_peak_vram = max(
            self.peak_vram_overall_mb,
            torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
        )

        ckpt_size_mb = 0.0
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)

        ply_size_mb = 0.0
        if ply_path and os.path.exists(ply_path):
            ply_size_mb = os.path.getsize(ply_path) / (1024 ** 2)

        summary_data = {
            "experiment_name": self.exp_name,
            "trainer_type": self.trainer_type,
            "result_dir": os.path.abspath(self.result_dir),
            "total_steps": final_step,
            "total_time_seconds": round(total_time_s, 2),
            "total_time_formatted": formatted_time,
            "average_fps": round(avg_fps, 2),
            "average_step_ms": round(avg_step_ms, 2),
            "peak_vram_mb": round(overall_peak_vram, 2),
            "final_vram_allocated_mb": round(final_vram_alloc, 2),
            "final_num_gs": final_num_gs,
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "ply_size_mb": round(ply_size_mb, 2),
            "final_metrics": self.last_eval_metrics,
            "config": config_dict or {},
        }

        # Lưu JSON
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # Lưu TXT
        txt_content = f"""========================================================================================
BÁO CÁO TỔNG KẾT HUẤN LUYỆN 3D GAUSSIAN SPLATTING
========================================================================================
- Tên thử nghiệm:            {self.exp_name}
- Phiên bản trainer:         {self.trainer_type.upper()}
- Thư mục kết quả:           {self.result_dir}
- Tổng số bước (Steps):      {final_step:,}
- Tổng thời gian:            {formatted_time} ({total_time_s:.1f}s)
- Tốc độ trung bình:         {avg_fps:.1f} steps/s ({avg_step_ms:.1f} ms/step)
----------------------------------------------------------------------------------------
BỘ NHỚ VRAM & MÔ HÌNH:
- Đỉnh VRAM (Peak VRAM):     {overall_peak_vram:.1f} MB
- VRAM tĩnh cuối cùng:       {final_vram_alloc:.1f} MB
- Số lượng Gaussians:        {final_num_gs:,}
- Kích thước Checkpoint:     {ckpt_size_mb:.2f} MB
- Kích thước PLY:            {ply_size_mb:.2f} MB
----------------------------------------------------------------------------------------
CHẤT LƯỢNG KẾT XUẤT (EVALUATION):
- PSNR:                      {self.last_eval_metrics.get('psnr', 'N/A')} dB
- SSIM:                      {self.last_eval_metrics.get('ssim', 'N/A')}
- LPIPS:                     {self.last_eval_metrics.get('lpips', 'N/A')}
========================================================================================
File log chi tiết:
  - Thông số bước:  {self.train_csv_path if not self.eval_only else 'N/A (Chế độ eval_only)'}
  - Thông số eval:  {self.eval_csv_path}
  - TensorBoard:    {os.path.join(self.result_dir, 'tb')}
========================================================================================
"""
        with open(self.summary_txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        print("\n" + txt_content)

        # Đóng các file CSV
        if self.train_csv_file:
            self.train_csv_file.close()
        if self.eval_csv_file:
            self.eval_csv_file.close()

        return summary_data
