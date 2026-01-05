import argparse
import configparser
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, TextIO

import cv2
import numpy as np
from tqdm import tqdm

CLASS_MAP_KEYWORDS: Dict[str, int] = {
    "player": 0, "goalkeeper": 0, "player team right": 0, "player team left": 0,
    "goalkeeper team right": 0, "goalkeeper team left": 0,
    "goalkeepers team left": 0, "goalkeepers team right": 0,
    "ball": 1,
    "referee": 2
}

def setup_logging():
    """Configures the logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def create_directory_link(target: Path, link: Path) -> None:
    """
    Creates a symbolic link or Directory Junction (Windows) to satisfy YOLO structure.
    
    YOLO expects an 'images' folder. SoccerNet uses 'img1'. 
    We create a link 'images' -> 'img1' so YOLO can infer the 'labels' path correctly.

    Args:
        target: The source directory (e.g., .../img1).
        link: The link name to create (e.g., .../images).
    """
    if link.exists():
        return

    try:
        if os.name == 'nt':
            os.system(f'mklink /J "{link}" "{target}" >nul 2>&1')
        else:
            os.symlink(target, link)
    except Exception as e:
        logging.warning(f"[ WARN | Failed to create link {link} -> {target}: {e} ]")

def load_ini_config(ini_path: Path) -> Optional[configparser.ConfigParser]:
    """
    Parses an INI configuration file.

    Args:
        ini_path: Path to the .ini file.

    Returns:
        ConfigParser object if successful, None otherwise.
    """
    try:
        config = configparser.ConfigParser()
        config.read(ini_path)
        return config
    except Exception:
        return None

def resolve_class_id(description: str) -> int:
    """
    Maps a raw SoccerNet class description to a YOLO class ID.

    Args:
        description: The class string from gameinfo.ini.

    Returns:
        int: Class ID (0, 1, 2) or -1 if unknown.
    """
    desc_lower = description.lower()
    for keyword, cls_id in CLASS_MAP_KEYWORDS.items():
        if keyword in desc_lower:
            return cls_id
    return -1

def get_sequence_dimensions(seq_path: Path, img1_dir: Path) ->tuple:
    """
    Retrieves image dimensions from seqinfo.ini or fallback to reading the first image.

    Args:
        seq_path: Path to the sequence directory.
        img1_dir: Path to the image directory.

    Returns:
        Tuple (width, height).
    """
    img_w, img_h = 1920, 1080 
    
    seqinfo_path = seq_path / "seqinfo.ini"
    if seqinfo_path.exists():
        cfg = load_ini_config(seqinfo_path)
        if cfg:
            try:
                img_w = int(cfg['Sequence']['imWidth'])
                img_h = int(cfg['Sequence']['imHeight'])
                return img_w, img_h
            except KeyError:
                pass
    
    first_img = next(img1_dir.glob("*.jpg"), None)
    if first_img:
        im = cv2.imread(str(first_img))
        if im is not None:
            img_h, img_w = im.shape[:2]
            
    return img_w, img_h

def process_sequence(seq_path: Path, list_file_handle: TextIO, subsample_rate: int = 5) -> None:
    """
    Process a single SoccerNet sequence: links folders, generates labels, updates manifest.

    Args:
        seq_path: Path to the sequence directory (e.g., SNMOT-001).
        list_file_handle: File handle for the dataset manifest text file.
        subsample_rate: Rate at which to sample frames (to avoid high correlation).
    """
    img1_dir = seq_path / "img1"
    labels_dir = seq_path / "labels"
    images_link = seq_path / "images" 

    if not img1_dir.exists():
        logging.debug(f"Skipping {seq_path.name}: img1 not found.")
        return

    create_directory_link(img1_dir, images_link)
    labels_dir.mkdir(exist_ok=True)

    img_w, img_h = get_sequence_dimensions(seq_path, img1_dir)

    id_to_class: Dict[int, int] = {}
    gameinfo_path = seq_path / "gameinfo.ini"
    
    if gameinfo_path.exists():
        g_cfg = load_ini_config(gameinfo_path)
        if g_cfg and 'Sequence' in g_cfg:
            for k, v in g_cfg['Sequence'].items():
                if k.startswith("trackletid_"):
                    try:
                        tid = int(k.split('_')[1])
                        cid = resolve_class_id(v)
                        if cid != -1:
                            id_to_class[tid] = cid
                    except ValueError:
                        continue

    gt_path = seq_path / "gt" / "gt.txt"
    if not gt_path.exists():
        return

    frame_data: Dict[int, list] = {}

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue

            if track_id not in id_to_class:
                continue
            
            cls_id = id_to_class[track_id]

            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h

            x_c = np.clip(x_c, 0, 1)
            y_c = np.clip(y_c, 0, 1)
            w_n = np.clip(w_n, 0, 1)
            h_n = np.clip(h_n, 0, 1)

            label_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n"

            if frame_idx not in frame_data:
                frame_data[frame_idx] = []
            frame_data[frame_idx].append(label_line)

    for frame_idx, lines in frame_data.items():
        file_name = f"{frame_idx:06d}.txt"
        with open(labels_dir / file_name, 'w') as f_out:
            f_out.writelines(lines)

    images = sorted(list(images_link.glob("*.jpg")))
    
    images = images[::subsample_rate]
    
    for img_path in images:
        list_file_handle.write(f"{img_path.resolve().as_posix()}\n")

def main():
    """Main execution entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Prepare SoccerNet Tracking Data for YOLO.")
    parser.add_argument("--root", type=str, default="data/datasets/SoccerNet/tracking/train", 
                        help="Root directory of the SoccerNet train set.")
    parser.add_argument("--output_list", type=str, default="soccernet_train_list.txt",
                        help="Output path for the dataset manifest file.")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Subsampling rate for frames (default: 5).")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.root)
    output_list_path = Path(args.output_list)

    if not dataset_root.exists():
        logging.error(f"[ ERROR | Dataset root not found: {dataset_root} ]")
        sys.exit(1)

    sequences = sorted([d for d in dataset_root.iterdir() if d.is_dir() and "SNMOT" in d.name])
    
    logging.info(f"[ INFO | Processing {len(sequences)} sequences from {dataset_root} ]")
    logging.info(f"[ INFO | In-Place operation: Creating 'images' links and 'labels' folders ]")

    with open(output_list_path, 'w') as list_file:
        for seq in tqdm(sequences, desc="Converting Sequences"):
            process_sequence(seq, list_file, args.subsample)

    logging.info(f"[ SUCCESS | Dataset prepared. Manifest saved to: {output_list_path.resolve()} ]")

if __name__ == "__main__":
    main()