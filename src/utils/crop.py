from __future__ import annotations
import argparse
import json
import random
import cv2
import yaml
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def parse_mot_line(line: str) -> Tuple[int, int, float, float, float, float, float]:
    parts = line.strip().split(",")
    if len(parts) < 7:
        raise ValueError(f"Bad line: {line!r}")
    frame = int(float(parts[0]))
    tid = int(float(parts[1]))
    x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
    conf = float(parts[6])
    return frame, tid, x, y, w, h, conf

def clamp_box(x, y, w, h, W, H):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

def find_frame(img_dir: Path, frame_id: int) -> Path:
    candidates = [
        img_dir / f"{frame_id:06d}.jpg",
        img_dir / f"{frame_id:06d}.png",
        img_dir / f"{frame_id}.jpg",
        img_dir / f"{frame_id}.png",
    ]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError(f"Frame {frame_id} not found in {img_dir}")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Worker Function
def process_frame_group(args):
    frame_path, crops_data, min_box_area = args
    
    # Carica immagine
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
            
        crop_img = img[y1:y1+ch, x1:x1+cw]
        
        if cv2.imwrite(str(dst_path), crop_img):
            written_count += 1
        else:
            skipped_count += 1
            
    return written_count, skipped_count

# Helper per dividere in chunk
def chunked_iterable(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

# ==========================================
# 🚀 MAIN
# ==========================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="YAML config path.")
    ap.add_argument("--workers", type=int, default=8, help="Num processi paralleli")
    ap.add_argument("--batch-size", type=int, default=500, help="Batch size per RAM")
    args = ap.parse_args()
    
    cfg = load_cfg(args.config)

    # 1. PATHS
    seq_root = Path(cfg["seq_root"])
    out_name = cfg.get("out_name", "SNTRACK_REID")
    img_subdir = cfg.get("img_subdir", "img1")
    gt_relpath = cfg.get("gt_relpath", "gt/gt.txt")

    # Params
    min_conf = float(cfg.get("min_conf", 1.0))
    min_box_area = int(cfg.get("min_box_area", 400))
    query_per_id = int(cfg.get("query_per_id", 1)) 
    # gallery_per_id viene ignorato: tutto il resto va in gallery
    seed = int(cfg.get("seed", 17))
    max_per_id = int(cfg.get("max_per_id", 0))

    # Output
    fastreid_datasets_root = Path("data/datasets/ReidCrop")
    fastreid_datasets_root.mkdir(parents=True, exist_ok=True)
    out_root = fastreid_datasets_root / out_name
    
    # Creiamo SOLO Query e Test (Gallery). Niente Train.
    query_dir = out_root / "query"
    test_dir = out_root / "bounding_box_test"
    
    # Se esiste la cartella train, la ignoriamo o la creiamo vuota per compatibilità
    # ma il codice sotto non ci scriverà nulla.
    train_dir_dummy = out_root / "bounding_box_train" 
    ensure_dir(query_dir); ensure_dir(test_dir); ensure_dir(train_dir_dummy)

    random.seed(seed)

    # 2. PIANIFICAZIONE
    print("Fase 1: Analisi Sequenze...")
    pid_map: Dict[Tuple[str, int], int] = {}
    next_pid = 0
    pid_recs: Dict[int, List[Tuple[Path, int, int, int, int, str]]] = {}

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    
    for seq in tqdm(seq_dirs, desc="Reading GTs"):
        seq_name = seq.name
        img_dir = seq / img_subdir
        gt_path = seq / gt_relpath
        if not gt_path.exists(): continue

        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: frame, tid, x, y, w, h, conf = parse_mot_line(line)
                except ValueError: continue
                if conf < min_conf: continue

                key = (seq_name, tid)
                if key not in pid_map:
                    pid_map[key] = next_pid
                    next_pid += 1
                pid = pid_map[key]
                
                try: frame_path = find_frame(img_dir, frame)
                except FileNotFoundError: continue
                
                dst_name = f"{pid:04d}_c0_f{frame:06d}_{seq_name}.jpg"
                pid_recs.setdefault(pid, []).append((frame_path, int(x), int(y), int(w), int(h), dst_name))

    # 3. SPLIT & TASK CREATION
    print(" Fase 2: Creazione Task (Split Query / Gallery)...")
    frame_tasks = defaultdict(list)

    for pid, recs in pid_recs.items():
        if max_per_id and max_per_id > 0 and len(recs) > max_per_id:
            random.shuffle(recs)
            recs = recs[:max_per_id]
        
        # Ordina per frame
        recs_sorted = sorted(recs, key=lambda r: r[5])
        
        # --- LOGICA DI SPLIT BINARIA ---
        # 1. Definisci quante immagini sono Query
        qn = min(query_per_id, len(recs_sorted))
        
        # 2. Slice delle liste
        query_list = recs_sorted[:qn]
        gallery_list = recs_sorted[qn:] # TUTTO IL RESTO va qui

        # 3. Assegnazione Cartelle
        for r in query_list: 
            frame_tasks[r[0]].append((r[1], r[2], r[3], r[4], query_dir / r[5]))
        
        for r in gallery_list: 
            frame_tasks[r[0]].append((r[1], r[2], r[3], r[4], test_dir / r[5]))

    all_tasks = [(fpath, crops, min_box_area) for fpath, crops in frame_tasks.items()]
    total_tasks = len(all_tasks)
    
    print(f" Fase 3: Esecuzione a Blocchi (Workers: {args.workers}, Batch: {args.batch_size})")
    print(f"   Frame totali da processare: {total_tasks}")
    
    total_written = 0
    total_skipped = 0
    
    # 4. ESECUZIONE SAFE (A BLOCCHI)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        
        pbar = tqdm(total=total_tasks, desc="Cropping Batches")
        
        for batch in chunked_iterable(all_tasks, args.batch_size):
            futures = [executor.submit(process_frame_group, t) for t in batch]
            
            for f in as_completed(futures):
                w, s = f.result()
                total_written += w
                total_skipped += s
                pbar.update(1)
            
            del futures
            gc.collect()
            
        pbar.close()

    # 5. METADATA
    meta = {
        "source": "SoccerNet Tracking MOT",
        "pids": next_pid,
        "crops_written": total_written,
        "skipped_small": total_skipped,
        "out_root": str(out_root),
        "split_type": "Query_vs_Gallery_Only"
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Completato! Output in: {out_root}")
    print(f"   Query Dir: {len(list(query_dir.glob('*.jpg')))} files")
    print(f"   Gallery Dir (Test): {len(list(test_dir.glob('*.jpg')))} files")
    print(f"   Train Dir: (Empty/Ignored)")

if __name__ == "__main__":
    main()