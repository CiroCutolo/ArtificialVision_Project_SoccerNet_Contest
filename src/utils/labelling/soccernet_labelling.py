import os
import configparser
import cv2
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURAZIONE ---
SOCCERNET_TRAIN_DIR = Path("data/datasets/SoccerNet/tracking/train")

# MAPPING (Stessa logica di prima)
CLASS_MAP_KEYWORDS = {
    "player": 0, "goalkeeper": 0, "player team right": 0, "player team left": 0,
    "goalkeeper team right": 0, "goalkeeper team left": 0,
    "goalkeepers team left": 0, "goalkeepers team right": 0,
    "ball": 1,
    "referee": 2
}

def create_junction(source, link_name):
    """
    Crea un Directory Junction su Windows (immagini -> img1).
    Non richiede permessi di amministratore come i symlink reali.
    """
    if link_name.exists():
        return # Già esiste

    try:
        # Usa os.symlink su Linux/Mac, o un comando CMD per Junction su Windows
        if os.name == 'nt':
            # mklink /J Link Target
            os.system(f'mklink /J "{link_name}" "{source}" >nul 2>&1')
        else:
            os.symlink(source, link_name)
    except Exception as e:
        print(f" Warning: Impossibile creare link {link_name}. YOLO potrebbe non trovare le immagini. {e}")

def parse_ini(ini_path):
    try:
        config = configparser.ConfigParser()
        config.read(ini_path)
        return config
    except: return None

def get_class_id(description):
    desc_lower = description.lower()
    for keyword, cls_id in CLASS_MAP_KEYWORDS.items():
        if keyword in desc_lower: return cls_id
    return -1

def process_sequence(seq_path, list_file_handle):
    img1_dir = seq_path / "img1"
    labels_dir = seq_path / "labels"
    images_link = seq_path / "images" # Il link fittizio per YOLO
    
    if not img1_dir.exists(): return

    # 1. Crea Junction images -> img1
    # YOLO cerca "/images/" nel path per sostituirlo con "/labels/". 
    # Senza questo link, YOLO cercherebbe le label dentro img1 (o fallirebbe).
    create_junction(img1_dir, images_link)

    # 2. Crea cartella labels (se non esiste)
    labels_dir.mkdir(exist_ok=True)

    # 3. Recupera Dimensioni (da seqinfo o fallback immagine)
    img_w, img_h = 1920, 1080
    seqinfo_path = seq_path / "seqinfo.ini"
    if seqinfo_path.exists():
        cfg = parse_ini(seqinfo_path)
        try:
            img_w = int(cfg['Sequence']['imWidth'])
            img_h = int(cfg['Sequence']['imHeight'])
        except: pass
    else:
        first = next(img1_dir.glob("*.jpg"), None)
        if first:
            im = cv2.imread(str(first))
            if im is not None: img_h, img_w = im.shape[:2]

    # 4. Mappa ID -> Classe
    id_to_class = {}
    gameinfo_path = seq_path / "gameinfo.ini"
    if gameinfo_path.exists():
        g_cfg = parse_ini(gameinfo_path)
        if g_cfg and 'Sequence' in g_cfg:
            for k, v in g_cfg['Sequence'].items():
                if k.startswith("trackletid_"):
                    try:
                        tid = int(k.split('_')[1])
                        cid = get_class_id(v)
                        if cid != -1: id_to_class[tid] = cid
                    except: pass

    # 5. Genera Label TXT frame per frame
    gt_path = seq_path / "gt" / "gt.txt"
    if not gt_path.exists(): return

    # Buffer per scrittura veloce
    frame_data = {}

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

            if track_id not in id_to_class: continue
            cls_id = id_to_class[track_id]

            # Normalizzazione YOLO
            x_c = (x + w/2) / img_w
            y_c = (y + h/2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            
            # Clip 0-1
            x_c, y_c = max(0, min(1, x_c)), max(0, min(1, y_c))
            w_n, h_n = max(0, min(1, w_n)), max(0, min(1, h_n))

            label_line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n"
            
            if frame_idx not in frame_data:
                frame_data[frame_idx] = []
            frame_data[frame_idx].append(label_line)

    # Scrivi su disco
    for frame_idx, lines in frame_data.items():
        # SoccerNet frames: 1 -> 000001.jpg
        file_name = f"{frame_idx:06d}.txt"
        with open(labels_dir / file_name, 'w') as f_out:
            f_out.writelines(lines)

    # 6. Aggiungi al Manifest (usando il path 'images' fittizio)
    # IMPORTANTE: Scriviamo il path che passa per il link 'images', non 'img1'
    # Così YOLO è felice.
    images = sorted(list(images_link.glob("*.jpg")))
    images = images[::5]
    for img_path in images:
        list_file_handle.write(f"{img_path.absolute().as_posix()}\n")

def main():
    sequences = sorted([d for d in SOCCERNET_TRAIN_DIR.iterdir() if d.is_dir() and "SNMOT" in d.name])
    list_file_path = Path("soccernet_train_list.txt")
    
    print(f" Elaborazione In-Place (con Junctions) di {len(sequences)} sequenze...")
    
    with open(list_file_path, 'w') as list_file:
        for seq in tqdm(sequences):
            process_sequence(seq, list_file)
            
    print("\n Fatto!")
    print(f"Dataset pronto in: {list_file_path.absolute()}")
    print("Nota: È stata creata una cartella 'labels' e un link 'images' in ogni SNMOT.")

if __name__ == "__main__":
    main()