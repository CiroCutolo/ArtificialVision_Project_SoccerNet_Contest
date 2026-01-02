import shutil
from pathlib import Path
from tqdm import tqdm

def remap_partition(source_dir: Path, target_dir: Path, class_mapping: dict) -> None:
    """
    Processes a single dataset partition (e.g., train or test), remapping class IDs
    and filtering out excluded classes.

    Args:
        source_dir (Path): The path to the folder containing original YOLO .txt labels.
        target_dir (Path): The path where the modified .txt labels will be saved.
        class_mapping (dict): A dictionary mapping {old_id: new_id}. 
                              Classes not in this dict are discarded.

    Returns:
        None

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if not source_dir.exists():
        print(f"[ERROR] Source partition not found: {source_dir}")
        raise FileNotFoundError(f"Directory {source_dir} does not exist.")

    target_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(source_dir.glob("*.txt"))
    print(f"[INFO] Processing {len(label_files)} files in {source_dir.name}...")

    processed_count = 0
    
    for file_path in tqdm(label_files):
        new_lines = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            try:
                original_id = int(parts[0])
            except ValueError:
                continue

            if original_id in class_mapping:
                new_id = class_mapping[original_id]
                coords = " ".join(parts[1:])
                new_line = f"{new_id} {coords}\n"
                new_lines.append(new_line)
        
        with open(target_dir / file_path.name, 'w') as f:
            f.writelines(new_lines)
            
        processed_count += 1

    print(f"[SUCCESS] Partition {source_dir.name} remapped. {processed_count} files generated.")


def process_soccana_remapping(root_path: str) -> None:
    """
    Orchestrates the label remapping for the Soccana dataset (Train and Test partitions).

    Structure assumptions:
    - Input: root_path/labels/train and root_path/labels/test
    - Output: root_path/labels/train_remapped and root_path/labels/test_remapped

    Args:
        root_path (str): The absolute path to the dataset root (e.g., .../Soccana/V1/V1).

    Returns:
        None
    """
    dataset_root = Path(root_path)
    labels_root = dataset_root / "labels"

    mapping = {0: 0, 2: 0}

    partitions = ["train", "test"]

    print(f"[INFO] Starting remapping for Soccana dataset at {dataset_root}")
    print(f"[INFO] Mapping Strategy: Player(0)->0, Referee(2)->0, Ball(1)->Drop")

    for partition in partitions:
        source = labels_root / partition
        target = labels_root / f"{partition}_remapped"
        
        if source.exists():
            remap_partition(source, target, mapping)
        else:
            print(f"[DEBUG] Partition {partition} not found, skipping.")

if __name__ == "__main__":
    soccana_root = "C:/Users/ciroc/Desktop/AV_project/data/datasets/Soccana/V1/V1"
    process_soccana_remapping(soccana_root)