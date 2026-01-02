import argparse
import logging
import glob
import re
import sys
import yaml
from pathlib import Path
from src.modules.behavior_analyzer import BehaviorAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BehaviorBatch")

class BatchProcessor:
    """Manages batch processing of existing tracking files to generate behavior counts."""

    def __init__(self, config_path: Path):
        """Initialize the Batch Processor.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self._load_yaml(config_path)
        self.analyzer = BehaviorAnalyzer(self.config)
        self.team_id = int(self.config["naming"]["team_id"])
        
    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load YAML configuration.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dictionary configuration.
        """
        logger.info(f"[ INFO | Loading Config: {path} ]")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _parse_tracking_file(path: Path) -> list:
        """Parse a MOT-format tracking file.

        Args:
            path: Path to the tracking text file.

        Returns:
            List of tuples (frame, x, y, w, h).
        """
        dets = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    if not line.strip(): continue
                    parts = [x.strip() for x in line.split(",")]
                    
                    if len(parts) < 6:
                        logger.warning(f"[ WARN | Line {ln} in {path.name} malformed ]")
                        continue
                        
                    frame = int(float(parts[0]))
                    x, y, w, h = map(float, parts[2:6])
                    dets.append((frame, x, y, w, h))
        except Exception as e:
            logger.error(f"[ ERROR | Failed to parse {path}: {e} ]")
        return dets

    def _extract_video_id(self, path: Path) -> int:
        """Extract sequence ID from filename or path.

        Args:
            path: File path object.

        Returns:
            Integer representing the video ID.
        """
        name = path.name
        m = re.search(r"tracking_(\d+)_\d+\.txt$", name)
        if m: return int(m.group(1))
        
        m = re.search(r"SNMOT-(\d+)", str(path).replace("\\", "/"))
        if m: return int(m.group(1))
        
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    def run(self, stage: str = "both"):
        """Execute the batch processing based on the selected stage.

        Args:
            stage: Processing mode ('pred', 'gt', or 'both').
        """
        patterns = []
        
        if stage in ["pred", "both"]:
            glob_pat = self.config["patterns"]["pred_tracking"]
            out_dir = Path(self.config["output"]["pred_behavior_dir"])
            patterns.append(("PRED", glob_pat, out_dir, "flat"))

        if stage in ["gt", "both"]:
            glob_pat = self.config["patterns"]["gt_tracking"]
            patterns.append(("GT", glob_pat, None, "inplace"))

        for mode, pattern, dest_dir, save_style in patterns:
            logger.info(f"[ INFO | Starting Batch: {mode} | Pattern: {pattern} ]")
            files = sorted(glob.glob(pattern))
            
            if not files:
                logger.warning(f"[ WARN | No files found for pattern: {pattern} ]")
                continue
                
            for file_path in files:
                t_path = Path(file_path)
                video_id = self._extract_video_id(t_path)
                
                detections = self._parse_tracking_file(t_path)
                
                self.analyzer.process_tracks(detections)
                
                if save_style == "inplace":
                    out_path = t_path.parent / f"behavior_{video_id}_{self.team_id:02d}.txt"
                else:
                    out_path = dest_dir / f"behavior_{video_id}_{self.team_id:02d}.txt"
                
                self.analyzer.save_results(out_path)
                logger.info(f"[ {mode} | Processed Video {video_id} -> {out_path.name} ]")

def main():
    parser = argparse.ArgumentParser(description="Generate Behavior Counts from Tracking Files")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument("--stage", choices=["pred", "gt", "both"], default="both", help="Which data to process")
    
    args = parser.parse_args()
    
    if not args.config.exists():
        logger.error(f"[ ERROR | Config file not found: {args.config} ]")
        sys.exit(1)
        
    try:
        processor = BatchProcessor(args.config)
        processor.run(args.stage)
        logger.info("=== BATCH PROCESSING COMPLETED ===")
    except Exception as e:
        logger.error(f"[ FATAL | Process failed: {e} ]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()