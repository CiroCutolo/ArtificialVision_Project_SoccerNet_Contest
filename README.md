# TITOLO: Multi-Object Tracking & Behavior Analysis for Soccer
### PROGETTO ARTIFICIAL VISION - A.A. 2025/2026
## STUDENTI: [Ciro Cutolo, Federica Celano] - MATRICOLA: [0622702532, 0622702581]

ISTRUZIONI PER L'INSTALLAZIONE E L'ESECUZIONE (SETUP DI GARA)

--------------------------------------------------------------------------------
## 1. REQUISITI DI SISTEMA
Le dipendenze necessarie sono elencate nel file 'requirements.txt'.

Per installare le dipendenze:
- pip install -r requirements.txt

## 2. ORGANIZZAZIONE CARTELLE
La cartella consegnata è strutturata come segue:

├── src/                
├── configs/        
├── data/  
├── requirements/           
├── README.txt/
└── setup.py

## 3. CONFIGURAZIONE
Prima di eseguire l'inferenza, è necessario indicare al sistema dove si trovano
le sequenze video di test.

1. Aprire il file: configs/config.yaml
2. Modificare la riga sotto la voce "paths" -> "dataset_root".
3. Inserire il percorso assoluto alla cartella che contiene le sottocartelle
   delle sequenze (SNMOT-149, SNMOT-150, ...).

Esempio nel config.yaml:
paths:
  dataset_root: "C:/.../SoccerNet/test"

Inoltre, sempre nel file di configurazione, è necessario specificare i pesi del detector e del ReID module, 
e il path dell'ouput dell'inferenza:
- paths:
    inference_output_path: "C:/.../AV_project/data/models/evaluation/soccana_1cls_640_smoothed"
...
- detection:
    model_path: "C:/.../AV_project/data/models/weights/detector/best_50_epochs_soccana_1cls.pt" 
...
- tracking:
    reid_model_path: "C:/.../AV_project/data/models/weights/tracker/osnet_x1_0_soccernet.pt"
  
## 4. ESECUZIONE INFERENZA
Una volta configurato il percorso nel file yaml, tramite terminale raggiungere la cartella in cui è situato il file d'inferenza ed eseguirlo:

    python ./inference.py

Il sistema:
1. Caricherà i pesi dai percorsi relativi in 'weights/'.
2. Leggerà le sequenze dal percorso indicato nel config.
3. Genererà i file di output nella cartella specificata.

## 5. FORMATO OUTPUT
I risultati saranno salvati conformemente alle specifiche di gara:
- Tracking: tracking_K_XX.txt (es. tracking_149_01.txt)
- Behavior: behavior_K_XX.txt (es. behavior_149_01.txt)
