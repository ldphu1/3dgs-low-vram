import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

# Khắc phục lỗi mã hóa tiếng Việt trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from gsplat import export_splats


def convert_checkpoint(
    ckpt_path: Union[str, Path, Sequence[Union[str, Path]]],
    out_dir: Union[str, Path],
    formats: Sequence[str] = ("splat", "ply_compressed"),
    output_name: Optional[str] = None,
) -> Dict[str, str]:
    """Convert training checkpoint(s) to viewable 3DGS formats (.splat, .ply, ply_compressed).

    Args:
        ckpt_path: Path to checkpoint .pt file or list of shard paths.
        out_dir: Output directory where converted files will be saved.
        formats: Formats to export ("splat", "ply_compressed", "ply").
        output_name: Optional base filename without extension. If None, derived from checkpoint.

    Returns:
        Dict mapping format name to absolute path of exported file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(ckpt_path, (str, Path)):
        paths = [Path(ckpt_path)]
    else:
        paths = [Path(p) for p in ckpt_path]

    if not paths:
        raise ValueError("No checkpoint paths provided.")

    # Load shard(s)
    shards = [torch.load(p, map_location="cpu") for p in paths]
    first_shard = shards[0]

    is_compressed = bool(
        first_shard.get("is_compressed", False)
        or (isinstance(first_shard.get("splats"), dict) and first_shard["splats"].get("is_compressed", False))
    )

    if len(shards) == 1:
        raw_splats = first_shard.get("splats", first_shard)
        splats = {k: v.float() if torch.is_tensor(v) else v for k, v in raw_splats.items()}
        sh_codebook = splats.get("sh_codebook", first_shard.get("sh_codebook", None))
        sh_indices = splats.get("sh_indices", first_shard.get("sh_indices", None))
    else:
        # Multi-rank shard concatenation
        all_keys = list(shards[0]["splats"].keys())
        splats = {}
        for k in all_keys:
            tensors = [s["splats"][k] for s in shards if k in s["splats"]]
            if tensors and torch.is_tensor(tensors[0]):
                splats[k] = torch.cat(tensors, dim=0).float()
        sh_codebook = first_shard.get("sh_codebook", None)
        indices_list = [s["sh_indices"] for s in shards if "sh_indices" in s]
        sh_indices = torch.cat(indices_list, dim=0) if indices_list else None

    means = splats["means"]
    # CRITICAL: export_splats expects raw log-scales and raw logit-opacities.
    # Do NOT pass torch.exp(scales) or torch.sigmoid(opacities), as export_splats
    # applies exp/sigmoid internally for the appropriate formats!
    scales = splats["scales"]
    quats = torch.nn.functional.normalize(splats["quats"], dim=-1)
    opacities = splats["opacities"].squeeze(-1) if splats["opacities"].ndim > 1 else splats["opacities"]
    sh0 = splats["sh0"]
    if sh0.ndim == 2:
        sh0 = sh0.unsqueeze(1)

    total_splats = means.shape[0]

    # Reconstruct higher-order SH (shN)
    if is_compressed and sh_codebook is not None and sh_indices is not None:
        shN = sh_codebook[sh_indices.long()].float()
    elif "shN" in splats and splats["shN"] is not None:
        shN = splats["shN"].float()
    else:
        shN = torch.zeros((total_splats, 0, 3), dtype=torch.float32)

    if output_name is None:
        base_name = paths[0].stem
        if base_name.endswith("_rank0"):
            base_name = base_name[:-6]
    else:
        base_name = output_name

    results = {}
    for fmt in formats:
        if fmt == "splat":
            filename = f"{base_name}.splat"
        elif fmt == "ply_compressed":
            filename = f"{base_name}_compressed.ply"
        elif fmt == "ply":
            filename = f"{base_name}.ply"
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from 'splat', 'ply_compressed', 'ply'.")

        save_path = str(out_dir / filename)
        export_splats(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            format=fmt,
            save_to=save_path,
        )
        results[fmt] = save_path

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert 3DGS checkpoint to splat/ply formats.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--output_name", type=str, default=None, help="Base output filename")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["splat", "ply_compressed"],
        choices=["splat", "ply_compressed", "ply"],
        help="Formats to export",
    )
    args = parser.parse_args()

    print(f"Đang đọc checkpoint từ: {args.ckpt}...")
    results = convert_checkpoint(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        formats=args.formats,
        output_name=args.output_name,
    )

    print("\n" + "=" * 70)
    print("HOÀN TẤT XUẤT CÁC ĐỊNH DẠNG:")
    for fmt, path in results.items():
        size_mb = os.path.getsize(path) / (1024**2)
        print(f" - [{fmt}] {path} ({size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
