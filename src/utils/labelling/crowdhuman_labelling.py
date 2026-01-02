import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CrowdHuman_Converter")

def convert_single_entry(data: Dict[str, Any], img_folder: Path, output_folder: Path) -> bool:
    """
    Processes a single CrowdHuman annotation entry and saves it in YOLO format.

    Args:
        data (Dict[str, Any]): The parsed JSON dictionary representing one image annotation.
        img_folder (Path): Directory containing the source images.
        output_folder (Path): Directory where the resulting .txt file will be saved.

    Returns:
        bool: True if the file was successfully processed and saved, False if the image was not found.
    """
    img_id = data['ID']
    img_filename = f"{img_id}.jpg"
    img_path = img_folder / img_filename

    if not img_path.exists():
        return False

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    yolo_lines = []

    for obj in data['gtboxes']:
        tag = obj['tag']
        
        if tag != 'person':
            continue
        
        if 'extra' in obj:
            extra = obj['extra']
            if extra.get('ignore', 0) == 1 or extra.get('unsure', 0) == 1:
                continue

        fbox = obj['fbox']
        x, y, w, h = fbox
        
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    out_path = output_folder / f"{img_id}.txt"
    with out_path.open('w') as out_f:
        if yolo_lines:
            out_f.write("\n".join(yolo_lines))
    
    return True

def convert_crowdhuman_to_yolo(odgt_path: str, img_folder: str, output_folder: str) -> None:
    """
    Orchestrates the conversion of the entire CrowdHuman dataset from ODGT to YOLO format.

    Args:
        odgt_path (str): Path to the annotation file (.odgt).
        img_folder (str): Path to the folder containing images.
        output_folder (str): Path where YOLO .txt labels will be generated.

    Raises:
        FileNotFoundError: If the ODGT file does not exist.
    """
    odgt_file = Path(odgt_path)
    img_dir = Path(img_folder)
    out_dir = Path(output_folder)

    if not odgt_file.exists():
        logger.error(f"[ ERROR | Annotation file not found: {odgt_file} ]")
        raise FileNotFoundError(f"File {odgt_file} not found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[ INFO | Loading annotations from {odgt_file} ]")

    with odgt_file.open('r') as f:
        lines = f.readlines()

    processed_count = 0
    skipped_count = 0

    for line in tqdm(lines, desc="Converting"):
        try:
            data = json.loads(line)
            success = convert_single_entry(data, img_dir, out_dir)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            logger.error(f"[ ERROR | Failed processing line: {e} ]")
            skipped_count += 1

    logger.info(f"[ SUCCESS | Conversion completed. Processed: {processed_count}, Skipped/Missing: {skipped_count} ]")
    logger.info(f"[ METRICS | Output directory: {out_dir} ]")

if __name__ == "__main__":
    # Configuration based on user context
    ODGT_PATH = 'annotation_val.odgt'
    IMG_FOLDER = 'C:/Users/ciroc/Desktop/AV_project/data/datasets/CrowdHuman/images/'
    OUTPUT_FOLDER = 'C:/Users/ciroc/Desktop/AV_project/data/datasets/CrowdHuman/labels/'
    
    convert_crowdhuman_to_yolo(ODGT_PATH, IMG_FOLDER, OUTPUT_FOLDER)