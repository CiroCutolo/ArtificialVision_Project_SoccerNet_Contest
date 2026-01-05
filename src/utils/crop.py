from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Generator, Optional

import cv2
import yaml
from tqdm import tqdm

CropInstruction = Tuple[int, int, int, int, Path]
FrameTask = Tuple[Path, List[CropInstruction], int]

def setup_logging() -> None:
    """Configures the global logging state."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

logger = logging.getLogger(__name__)

def clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Clamps bounding box coordinates to ensure they fit within image boundaries.

    Args:
        x, y, w, h: Floating point coordinates from tracking data.
        img_w, img_h: Image dimensions.

    Returns:
        Tuple (x1, y1, width, height) integers, clamped.
    """
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

def parse_mot_line(line: str) -> Tuple[int, int, float, float, float, float, float]:
    """
    Parses a single line from a MOT-format GT file.
    Format: frame, id, left, top, width, height, conf, ...

    Returns:
        frame_idx, track_id, x, y, w, h, confidence
    """
    parts = line.strip().split(",")
    if len(parts) < 7:
        raise ValueError(f"Insufficient columns in line: {line!r}")
    
    frame = int(float(parts[0]))
    tid = int(float(parts[1]))
    x = float(parts[2])
    y = float(parts[3])
    w = float(parts[4])
    h = float(parts[5])
    conf = float(parts[6])
    
    return frame, tid, x, y, w, h, conf

def find_image_file(img_dir: Path, frame_id: int) -> Path:
    """
    Locates an image file for a given frame ID, checking common extensions.
    
    Args:
        img_dir: Directory containing images.
        frame_id: The integer frame number.

    Returns:
        Path object to the image.

    Raises:
        FileNotFoundError: If the image cannot be found.
    """
    candidates = [
        f"{frame_id:06d}.jpg",
        f"{frame_id:06d}.png",
        f"{frame_id}.jpg",
        f"{frame_id}.png",
    ]
    
    for name in candidates:
        p = img_dir / name
        if p.exists():
            return p
            
    raise FileNotFoundError(f"Frame {frame_id} not found in {img_dir}")

def chunked_iterable(iterable: List[Any], size: int) -> Generator[List[Any], None, None]:
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

def process_frame_batch(args: FrameTask) -> Tuple[int, int]:
    """
    Worker function to process crops for a single frame.
    Designed to minimize I/O by reading the image once and performing multiple crops.

    Args:
        args: Tuple containing (image_path, list_of_crops, min_area).

    Returns:
        (written_count, skipped_count)
    """
    frame_path, crops_data, min_box_area = args
    
    img = cv2.imread(str(frame_path))
    if img is None:
        return 0, 0
    
    H, W = img.shape[:2]
    written_count = 0
    skipped_count = 0
    
    for (x, y, w, h, dst_path) in crops_data:
        x1, y1, cw, ch = clamp_box(x, y, w, h, W, H)
        
        if cw * ch < min_box_area:
            skipped_count += 1
            continue
            
        crop_img = img[y1 : y1 + ch, x1 : x1 + cw]
        
        if crop_img.size == 0:
            skipped_count += 1
            continue

        if cv2.imwrite(str(dst_path), crop_img):
            written_count += 1
        else:
            skipped_count += 1
            
    return written_count, skipped_count

def main() -> None:
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Prepare ReID Dataset from Tracking GT.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes.")
    parser.add_argument("--batch-size", type=int, default=500, help="Task batch size for memory management.")
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seq_root = Path(cfg["seq_root"])
    out_name = cfg.get("out_name", "SNTRACK_REID")
    img_subdir = cfg.get("img_subdir", "img1")
    gt_relpath = cfg.get("gt_relpath", "gt/gt.txt")
    
    min_conf = float(cfg.get("min_conf", 1.0))
    min_box_area = int(cfg.get("min_box_area", 400))
    query_per_id = int(cfg.get("query_per_id", 1))
    seed = int(cfg.get("seed", 42))
    max_per_id = int(cfg.get("max_per_id", 0))

    out_root = Path("C:/Users/ciroc/Desktop/AV_project/data/datasets/ReidCrop") / out_name
    query_dir = out_root / "query"
    test_dir = out_root / "bounding_box_test"
    train_dir = out_root / "bounding_box_train" 
    
    for d in [query_dir, test_dir, train_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source: {seq_root}")
    logger.info(f"Target: {out_root}")

    random.seed(seed)

    logger.info("Scanning sequences and parsing GT...")

    mode = cfg.get("dataset_mode", "test").lower()
    
    pid_map: Dict[Tuple[str, int], int] = {}
    next_pid = 0
    pid_records: Dict[int, List[Tuple[Path, int, int, int, int, str]]] = defaultdict(list)

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    
    for seq in tqdm(seq_dirs, desc="Parsing GTs"):
        seq_name = seq.name
        img_dir = seq / img_subdir
        gt_path = seq / gt_relpath
        
        if not gt_path.exists():
            continue

        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                
                try:
                    frame, tid, x, y, w, h, conf = parse_mot_line(line)
                except ValueError:
                    continue
                
                if conf < min_conf:
                    continue

                key = (seq_name, tid)
                if key not in pid_map:
                    pid_map[key] = next_pid
                    next_pid += 1
                pid = pid_map[key]
                
                try:
                    frame_path = find_image_file(img_dir, frame)
                except FileNotFoundError:
                    continue
                
                dst_name = f"{pid:04d}_c0_f{frame:06d}_{seq_name}.jpg"
                pid_records[pid].append((frame_path, int(x), int(y), int(w), int(h), dst_name))

    logger.info(f"Found {next_pid} unique identities. Preparing split tasks...")

    frame_tasks: Dict[Path, List[CropInstruction]] = defaultdict(list)

    for pid, records in pid_records.items():
        if max_per_id > 0 and len(records) > max_per_id:
            random.shuffle(records)
            records = records[:max_per_id]

        records_sorted = sorted(records, key=lambda r: r[5])

        if mode == "train":
            
            for r in records_sorted[::25]:
                 frame_tasks[r[0]].append((r[1], r[2], r[3], r[4], train_dir / r[5]))
                 
        else:
            if max_per_id > 0 and len(records_sorted) > max_per_id:
                pass 

            records_subsampled = records_sorted[::25]

            qn = min(query_per_id, len(records_subsampled))
            query_list = records_subsampled[:qn]
            gallery_list = records_subsampled[qn:]

            for r in query_list:
                frame_tasks[r[0]].append((r[1], r[2], r[3], r[4], query_dir / r[5]))
                
            for r in gallery_list:
                frame_tasks[r[0]].append((r[1], r[2], r[3], r[4], test_dir / r[5]))

    all_tasks = [(fpath, crops, min_box_area) for fpath, crops in frame_tasks.items()]
    total_tasks = len(all_tasks)

    logger.info(f"Starting execution with {args.workers} workers. Total Frames: {total_tasks}")
    
    total_written = 0
    total_skipped = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=total_tasks, desc="Cropping Images", unit="frame")
        
        for batch in chunked_iterable(all_tasks, args.batch_size):
            futures = [executor.submit(process_frame_batch, t) for t in batch]
            
            for f in as_completed(futures):
                w, s = f.result()
                total_written += w
                total_skipped += s
                pbar.update(1)

            del futures
            gc.collect()
            
        pbar.close()

    meta = {
        "source": "SoccerNet Tracking MOT",
        "num_identities": next_pid,
        "total_crops": total_written,
        "skipped_small_area": total_skipped,
        "split_config": {
            "query_per_id": query_per_id,
            "min_conf": min_conf,
            "min_box_area": min_box_area
        }
    }
    
    with (out_root / "dataset_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    logger.info(" Dataset Generation Complete.")
    logger.info(f" Output Location: {out_root}")
    logger.info(f"  - Query Images:   {len(list(query_dir.glob('*.jpg')))}")
    logger.info(f"  - Gallery Images: {len(list(test_dir.glob('*.jpg')))}")
    logger.info(f"  - Train Images:   0 (Ignored)")

if __name__ == "__main__":
    main()