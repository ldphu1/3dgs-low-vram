"""
========================================================================================
SCRIPT TỰ ĐỘNG CHUẨN HÓA DATASET TANKS & TEMPLES (TANDT) CHO GSPLAT / SIMPLE_TRAINER
========================================================================================
Script này thực hiện:
1. Kiểm tra cấu trúc thư mục COLMAP (sparse/0/cameras.bin, images.bin, points3D.bin).
2. Tự động sinh các thư mục ảnh thu nhỏ (downsampled):
   - images_2 (giảm 2 lần độ phân giải)
   - images_4 (giảm 4 lần độ phân giải - tối ưu nhất cho GPU RTX 3050 4GB)
   - images_8 (giảm 8 lần độ phân giải - siêu nhẹ để test nhanh)
3. Hỗ trợ xử lý đa luồng (multithreading) cực nhanh.
4. Kiểm tra tương thích trực tiếp với Parser của gsplat.
========================================================================================
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

# Đảm bảo UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def downsample_single_image(args):
    src_path, dst_path, factor = args
    if os.path.exists(dst_path):
        return
    try:
        with Image.open(src_path) as img:
            new_w = max(1, img.width // factor)
            new_h = max(1, img.height // factor)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized.save(dst_path, quality=95)
    except Exception as e:
        print(f"Lỗi khi resize ảnh {src_path}: {e}")


def format_scene(scene_dir: str, factors=(2, 4, 8)):
    scene_path = Path(scene_dir).resolve()
    print("\n" + "=" * 80)
    print(f"ĐANG XỬ LÝ CẢNH: {scene_path.name.upper()} ({scene_path})")
    print("=" * 80)

    # 1. Kiểm tra thư mục images gốc
    images_dir = scene_path / "images"
    if not images_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục ảnh gốc tại: {images_dir}")
        return False

    # 2. Kiểm tra cấu trúc sparse / COLMAP
    sparse_0 = scene_path / "sparse" / "0"
    sparse_root = scene_path / "sparse"
    
    if not sparse_0.exists():
        # Nếu file bin nằm trực tiếp ở sparse thay vì sparse/0, tự động gom vào sparse/0
        sparse_bin_files = list(sparse_root.glob("*.bin")) + list(sparse_root.glob("*.txt"))
        if sparse_bin_files:
            print("Phát hiện file COLMAP nằm ở 'sparse/', đang chuyển vào 'sparse/0/'...")
            sparse_0.mkdir(parents=True, exist_ok=True)
            for f in sparse_bin_files:
                f.rename(sparse_0 / f.name)

    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    alt_files = ["cameras.txt", "images.txt", "points3D.txt"]
    has_bin = all((sparse_0 / f).exists() for f in required_files)
    has_txt = all((sparse_0 / f).exists() for f in alt_files)

    if not (has_bin or has_txt):
        print(f"[LỖI] Thiếu file COLMAP trong {sparse_0}! Cần có {required_files}")
        return False
    print(f"✓ Cấu trúc COLMAP hợp lệ: {sparse_0}")

    # 3. Lấy danh sách ảnh gốc
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in images_dir.iterdir() if f.suffix in image_extensions]
    num_images = len(image_files)
    print(f"✓ Tìm thấy {num_images} ảnh gốc trong 'images/'.")

    # 4. Sinh các thư mục images_2, images_4, images_8
    for factor in factors:
        out_dir = scene_path / f"images_{factor}"
        out_dir.mkdir(exist_ok=True)
        
        existing = len([f for f in out_dir.iterdir() if f.suffix in image_extensions])
        if existing == num_images:
            print(f"✓ Thư mục images_{factor}/ đã có đủ {existing} ảnh (bỏ qua).")
            continue

        print(f">> Đang tạo images_{factor}/ (giảm độ phân giải {factor}x bằng đa luồng)...")
        tasks = []
        for img_path in image_files:
            dst_path = out_dir / img_path.name
            tasks.append((str(img_path), str(dst_path), factor))

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
            list(tqdm(executor.map(downsample_single_image, tasks), total=len(tasks), desc=f"images_{factor}"))

    # 5. Kiểm tra tương thích với gsplat Parser
    try:
        sys.path.append(r"C:\Users\Asus\Desktop\3dgs\gsplat_repo\examples")
        from datasets.colmap import Parser
        parser = Parser(str(scene_path), factor=4)
        print(f"✓ [THÀNH CÔNG] gsplat Parser đã nạp thử thành công: {len(parser.image_names)} ảnh!")
    except Exception as e:
        print(f"[CẢNH BÁO] Kiểm tra Parser: {e}")

    return True


def main():
    base_data_dir = Path(r"C:\Users\Asus\Desktop\3dgs\data")
    if not base_data_dir.exists():
        print(f"Không tìm thấy thư mục: {base_data_dir}")
        return

    # Danh sách các cảnh tìm thấy
    scenes_to_process = []
    
    # Cảnh trong thư mục tandt (truck, train)
    tandt_dir = base_data_dir / "tandt"
    if tandt_dir.exists():
        for d in tandt_dir.iterdir():
            if d.is_dir() and (d / "images").exists():
                scenes_to_process.append(d)

    # Cảnh trong thư mục db (drjohnson, playroom)
    db_dir = base_data_dir / "db"
    if db_dir.exists():
        for d in db_dir.iterdir():
            if d.is_dir() and (d / "images").exists():
                scenes_to_process.append(d)

    if not scenes_to_process:
        print(f"Không tìm thấy cảnh COLMAP nào trong {base_data_dir}")
        return

    print("=" * 80)
    print(f"TÌM THẤY {len(scenes_to_process)} CẢNH CẦN CHUẨN HÓA:")
    for s in scenes_to_process:
        print(f" - {s}")
    print("=" * 80)

    for scene in scenes_to_process:
        format_scene(str(scene), factors=(2, 4))

    print("\n" + "=" * 80)
    print("HOÀN TẤT CHUẨN HÓA! BÂY GIỜ BẠN CÓ THỂ CHẠY HUẤN LUYỆN VỚI LỆNH:")
    print("=" * 80)
    truck_dir = r"C:\Users\Asus\Desktop\3dgs\data\tandt\truck"
    print(
        f'python C:\\Users\\Asus\\Desktop\\3dgs\\gsplat_repo\\examples\\simple_trainer.py default '
        f'--data_dir "{truck_dir}" '
        f'--result_dir "C:\\Users\\Asus\\Desktop\\3dgs\\results\\truck" '
        f'--data_factor 4 --disable_viewer'
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
