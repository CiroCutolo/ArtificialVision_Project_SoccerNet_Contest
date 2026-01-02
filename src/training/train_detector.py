import logging
from typing import Any, Optional
import torch
from ultralytics import YOLO
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AV_project_trainer")

class YOLOTrainer:
    """Wrapper for ultralytics.YOLO to manage model loading, training and validation."""

    def __init__(self, model_name: str):
        """Initialize trainer with a model identifier or path.

        Args:
            model_name: Path to weights file or model spec (e.g. 'yolov8n.pt').
        """
        self.model_name = model_name
        self.model = None

    def load_model(self, weights_path: Optional[str] = None):
        """Load a YOLO model into memory.

        The method prefers weights_path when provided, otherwise uses the model_name
        given at construction. The loaded model is stored on self.model and returned.

        Args:
            weights_path: Optional explicit path to weights or model spec.

        Returns:
            The instantiated YOLO model.
        """
        target = weights_path if weights_path else self.model_name
        logger.info(f"[ DEBUG | Loading Trainer Model: {target} ]")
        self.model = YOLO(target)
        return self.model

    def train(self, 
              data_yaml: str, 
              project_name: str, 
              run_name: str, 
              epochs: int, 
              imgsz: int, 
              batch_size: int, 
              hyp_yaml: str,
              device: int = 0,
              **kwargs) -> Any:
        """Train the loaded model.

        This method ensures a model is loaded, clears CUDA cache, then calls
        the underlying YOLO.train API with the provided parameters.

        Args:
            data_yaml: Path to dataset YAML (train/val paths and class names).
            project_name: Directory under which to save the run results.
            run_name: Subdirectory name for this run.
            epochs: Number of training epochs.
            imgsz: Input image size in pixels.
            batch_size: Batch size per device.
            hyp_yaml: Path to hyperparameters YAML or config accepted by YOLO.
            device: CUDA device index (or string like '0'/'cpu').

        Returns:
            The training results object returned by YOLO.train.
        """
        if self.model is None:
            self.load_model()
            
        torch.cuda.empty_cache()
        logger.info(f"[ DEBUG | Starting Training: {run_name} | ImgSz: {imgsz} | Batch: {batch_size} ]")
        logger.info(f"[ DEBUG | Using Hyperparameters: {hyp_yaml} ]")
        
        try:
            results = self.model.train(
                data=data_yaml,
                project=project_name,
                name=run_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                cfg=hyp_yaml,
                #classes=[0],
                close_mosaic=10,
                
                #single_cls=True,
                exist_ok=True,
                save=True,
                pretrained=True,
                plots=True,
                verbose=True,
                amp=True,

                **kwargs
            )
            logger.info(f"[ DEBUG | Training {run_name} completed successfully. ]")
            return results
        except Exception as e:
            logger.error(f"[ ERROR | Training crashed: {e} ]")
            raise e

    def validate(self, data_yaml: str, split: str = 'val') -> Any:
        """Run validation on the specified dataset split.

        Args:
            data_yaml: Path to dataset YAML used for validation.
            split: Dataset split to validate on (default 'val').

        Returns:
            Validation results from YOLO.val or None if model not loaded.
        """
        if self.model is None:
            logger.error("[ ERROR | Model not loaded for validation ]")
            return None
            
        logger.info(f"[ DEBUG | Validating on {split} set... ]")
        return self.model.val(data=data_yaml, split=split)
    
def main():
    trainer = YOLOTrainer(model_name="C:/Users/ciroc/Desktop/AV_project/data/models/weights/detector/best_50_epochs_crowdhuman_unfrozen_640.pt")
    trainer.load_model()
    trainer.train(
        data_yaml="C:/Users/ciroc/Desktop/AV_project/data/datasets/Soccana/V1/V1/data.yaml",
        hyp_yaml="C:/Users/ciroc/Desktop/AV_project/configs/params.yaml",
        project_name="runs/detector/training_soccana_1class",
        run_name="soccana_1class",
        epochs=50,

        imgsz=640,
        rect=False,
        patience=10,
        batch_size=8,
        device=0,
        workers=8,
        freeze=0,
        cache=False,
        warmup_epochs=3.0
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()