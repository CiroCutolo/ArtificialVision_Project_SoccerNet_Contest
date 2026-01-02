import torch
import torchreid
import os
import sys

# ================= CONFIGURAZIONE =================
# Sostituisci con il nome del file più recente che hai (es. il -2)
INPUT_CHECKPOINT = "model.pth.tar-2" 

# Architettura usata
MODEL_ARCH = 'osnet_x1_0'

# Output
OUTPUT_DIR = "exported_models"
# ==================================================

def main():
    if not os.path.exists(INPUT_CHECKPOINT):
        print(f"Errore: Il file {INPUT_CHECKPOINT} non esiste.")
        return

    print(f"Caricamento checkpoint: {INPUT_CHECKPOINT}...")
    
    # 1. Carica il checkpoint su CPU
    checkpoint = torch.load(INPUT_CHECKPOINT, map_location='cpu', weights_only=False)
    
    # Gestione caso in cui il checkpoint contenga tutto lo stato (optimizer, epoch, etc.)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"   -> Trovata chiave 'state_dict' (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint
        print("   -> Checkpoint è un raw state_dict.")

    # 2. Pulizia delle chiavi (rimozione prefisso 'module.' aggiunto da DataParallel)
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        clean_state_dict[new_k] = v
    
    # 3. Auto-detection del numero di classi (Cruciale per inizializzare il modello)
    # Cerchiamo il layer 'classifier.weight' per capire quanti ID c'erano
    if 'classifier.weight' in clean_state_dict:
        num_classes = clean_state_dict['classifier.weight'].shape[0]
        print(f"Rilevati {num_classes} ID (classi) dai pesi.")
    else:
        print("Impossibile rilevare num_classes. Defaulting a 1000 (rischio mismatch).")
        num_classes = 1000

    # 4. Build del modello vuoto con le dimensioni corrette
    print(f"Costruzione architettura {MODEL_ARCH}...")
    model = torchreid.models.build_model(
        name=MODEL_ARCH,
        num_classes=num_classes,
        pretrained=False # Non scaricare pesi imagenet, usiamo i nostri
    )
    
    # 5. Caricamento pesi
    try:
        model.load_state_dict(clean_state_dict)
        print("Pesi caricati correttamente nel modello.")
    except Exception as e:
        print(f"Errore caricamento state_dict: {e}")
        return

    model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================= EXPORT 1: Pesi Pytorch Puliti (.pt) =================
    pt_path = os.path.join(OUTPUT_DIR, f"{MODEL_ARCH}_soccernet.pt")
    torch.save(clean_state_dict, pt_path)
    print(f"\n[PYTHON] Pesi salvati in: {pt_path}")
    print("   (Usa questo per caricare in script Python puri)")

    # ================= EXPORT 2: TorchScript (.torchscript.pt) =============
    # Questo serve per C++ / BoT-SORT veloce
    ts_path = os.path.join(OUTPUT_DIR, f"{MODEL_ARCH}_soccernet.torchscript.pt")
    
    dummy_input = torch.randn(1, 3, 256, 128)
    
    try:
        # Tracing su CPU va benissimo per l'export
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        print(f"[C++ / DEPLOY] TorchScript salvato in: {ts_path}")
        print("   (Usa questo per TensorRT o integrazione C++)")
    except Exception as e:
        print(f"Errore durante il Tracing JIT: {e}")

if __name__ == "__main__":
    main()