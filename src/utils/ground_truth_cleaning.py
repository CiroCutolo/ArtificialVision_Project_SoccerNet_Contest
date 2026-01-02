import os
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm
import logging

# Configurazione Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GT_Cleaner_Metadata")

def get_ids_to_remove(ini_path: Path, target_classes: list = ["ball"]) -> set:
    """
    Legge gameinfo.ini e restituisce un set di ID (interi) che corrispondono
    alle classi da rimuovere (es. 'ball').
    
    Analizza righe tipo: trackletID_16= ball;none
    """
    ids_to_remove = set()
    
    # ConfigParser di Python richiede una sezione header (es. [Section]), 
    # ma a volte i file .ini custom non sono standard. 
    # Leggiamo il file riga per riga per massima robustezza.
    try:
        with open(ini_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Cerchiamo righe che iniziano con "trackletID_"
            if line.startswith("trackletID_") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()   # es. "trackletID_16"
                value = value.strip().lower() # es. "ball;none"
                
                # Estraiamo l'ID numerico dalla chiave
                # trackletID_16 -> 16
                try:
                    obj_id_str = key.replace("trackletID_", "")
                    obj_id = int(obj_id_str)
                    
                    # Controlliamo se il valore contiene una delle classi target
                    # es. se target è "ball" e value è "ball;none" -> MATCH
                    for target in target_classes:
                        if value.startswith(target) or target in value.split(';'):
                            ids_to_remove.add(obj_id)
                            break
                            
                except ValueError:
                    continue
                    
    except Exception as e:
        logger.warning(f"Errore lettura {ini_path}: {e}")
        
    return ids_to_remove

def clean_dataset_by_metadata(dataset_root: str):
    """
    Scansiona le cartelle, legge gameinfo.ini per identificare gli ID delle palle
    e pulisce i corrispettivi gt.txt.
    """
    root_path = Path(dataset_root)
    
    # Cerchiamo tutte le sequenze che hanno un gameinfo.ini
    ini_files = list(root_path.rglob("gameinfo.ini"))
    
    logger.info(f"Trovate {len(ini_files)} sequenze con metadati.")
    
    files_modified = 0
    total_removed_lines = 0

    for ini_path in tqdm(ini_files, desc="Processing Sequences"):
        seq_dir = ini_path.parent
        gt_path = seq_dir / "gt" / "gt.txt"
        
        if not gt_path.exists():
            continue
            
        # 1. Identifica gli ID da rimuovere (Palla)
        # Se vuoi rimuovere anche gli arbitri, aggiungi "referee" alla lista: ["ball", "referee"]
        ball_ids = get_ids_to_remove(ini_path, target_classes=["ball"])
        
        if not ball_ids:
            continue # Nessuna palla in questa sequenza (strano, ma possibile)

        # 2. Backup del file originale
        backup_path = gt_path.with_suffix(".txt.original_full")
        if not backup_path.exists():
            shutil.copy(gt_path, backup_path)
        
        # 3. Filtraggio GT
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        removed_count = 0
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            parts = line_str.split(',')
            
            try:
                # MOT Format: frame, ID, x, y, w, h, ...
                # L'ID è la seconda colonna (indice 1)
                obj_id = int(float(parts[1]))
                
                if obj_id in ball_ids:
                    removed_count += 1
                    continue # SALTA (Rimuovi)
                
                new_lines.append(line)
                
            except (IndexError, ValueError):
                # Se la riga è illeggibile, la teniamo per sicurezza
                new_lines.append(line)

        # 4. Scrittura solo se abbiamo rimosso qualcosa
        if removed_count > 0:
            with open(gt_path, 'w') as f:
                f.writelines(new_lines)
            
            files_modified += 1
            total_removed_lines += removed_count
            # logger.info(f"Seq {seq_dir.name}: Rimossi ID {ball_ids} ({removed_count} righe)")

    logger.info("-" * 40)
    logger.info(f"Pulizia Completata.")
    logger.info(f"Sequenze modificate: {files_modified}")
    logger.info(f"Totale righe (palle) rimosse: {total_removed_lines}")
    logger.info(f"Backup salvati come: .gt.txt.original_full")

if __name__ == "__main__":
    # IMPOSTA IL PATH CORRETTO
    DATASET_DIR = r"C:/Users/ciroc/Desktop/AV_project/data/datasets/SoccerNet/tracking/test1"
    
    clean_dataset_by_metadata(DATASET_DIR)