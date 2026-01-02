import logging
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from modules.detector import PlayerDetector
from modules.tracker import PlayerTracker
from modules.smoother import OnlineBoxSmoother
from modules.behavior_analyzer import BehaviorAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AV_Project_Inference")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"[ ERROR | Config file not found: {path} ]")
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"[ DEBUG | Configuration loaded from {path} ]")
    return cfg

def extract_sequence_id(seq_name: str) -> str:
    """Extract the numeric sequence ID from the folder name.
    
    Args:
        seq_name: The name of the sequence folder (e.g., 'SNMOT-149').

    Returns:
        The numeric string ID (e.g., '149').
    """
    if "SNMOT-" in seq_name:
        return seq_name.replace("SNMOT-", "")
    return seq_name

def run_inference():
    """Execute the full inference pipeline: Detection, Tracking, and Behavior Analysis.

    This function iterates over all sequences in the input directory, performs
    YOLO detection and BoT-SORT tracking frame-by-frame, updates the ROI
    counters, and saves both tracking (MOT format) and behavior (CSV format)
    results to the output directory defined in the config.
    """
    CONFIG_PATH = "C:/Users/ciroc/Desktop/AV_project/configs/config.yaml"
    
    cfg = load_config(CONFIG_PATH)

    dataset_root = Path(cfg["paths"]["dataset_root"]) if "paths" in cfg else Path(r"C:/Users/ciroc/Desktop/AV_project/data/datasets/SoccerNet")
    input_dir = dataset_root / "tracking/test" 
    output_dir = Path("C:/Users/ciroc/Desktop/AV_project/data/models/evaluation/soccana_1cls_640_smoothed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    team_id = int(cfg["naming"]["team_id"])
    
    detector = PlayerDetector(cfg)
    
    
    sequences = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    logger.info(f"[ INFO | Starting Inference on {len(sequences)} sequences ]")
    logger.info(f"[ INFO | Output Directory: {output_dir} ]")

    for seq_dir in tqdm(sequences, desc="Sequences"):
        box_smoother = OnlineBoxSmoother(alpha=0.7)
        seq_name = seq_dir.name
        seq_id = extract_sequence_id(seq_name)
        
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            img_dir = seq_dir / "images"
        
        if not img_dir.exists():
            logger.warning(f"[ WARN | Image directory not found for {seq_name}. Skipping. ]")
            continue

        tracking_out_file = output_dir / f"tracking_{seq_id}_{team_id:02d}.txt"
        behavior_out_file = output_dir / f"behavior_{seq_id}_{team_id:02d}.txt"

        tracker = PlayerTracker(cfg)
        behavior_engine = BehaviorAnalyzer(cfg)
        
        frame_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        tracking_buffer = []

        for img_path in frame_files:
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            raw_detections = detector.detect(frame)
            
            tracks = tracker.update(raw_detections, frame)
            tracks = box_smoother.update(tracks)

            for t in tracks:
                t_x1, t_y1, t_x2, t_y2 = t[:4]
                t_id = int(t[4])
                t_conf = t[5]
                
                width = t_x2 - t_x1
                height = t_y2 - t_y1
                
                behavior_engine.update(frame_idx, t_x1, t_y1, width, height)

                line = f"{frame_idx},{t_id},{t_x1:.2f},{t_y1:.2f},{width:.2f},{height:.2f},{t_conf:.2f},-1,-1,-1"
                tracking_buffer.append(line)

        if tracking_buffer:
            with tracking_out_file.open("w") as f:
                f.write("\n".join(tracking_buffer))
            logger.info(f"[ DEBUG | Saved Tracking: {tracking_out_file.name} ]")
        else:
            tracking_out_file.touch()
            logger.warning(f"[ WARN | Created empty tracking file for {seq_name} ]")

        behavior_engine.save_results(behavior_out_file)
        logger.info(f"[ DEBUG | Saved Behavior: {behavior_out_file.name} ]")

    logger.info("[ SUCCESS | Inference Procedure Completed ]")

if __name__ == "__main__":
    run_inference()