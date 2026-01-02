import sys
import os
import logging
import torch
import torchreid
from torchreid import utils
from pathlib import Path
from typing import Optional, Dict, Any
from src.modules.soccernet_reid_dataset import SoccerNetReIDDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AV_project_reid_trainer")

class ReIDTrainer:
    """Wrapper for torchreid to manage ReID model setup, training, and export."""

    def __init__(self, 
                 dataset_path: str, 
                 save_dir: str, 
                 model_arch: str = 'osnet_x1_0',
                 device_id: int = 0):
        """Initialize the ReID trainer configurations.

        Args:
            dataset_path: Path to the root of the ReID dataset (crops).
            save_dir: Directory where logs and checkpoints will be saved.
            model_arch: Architecture name supported by torchreid (default: 'osnet_x1_0').
            device_id: GPU device index.
        """
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.model_arch = model_arch
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        self.datamanager = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.engine = None

        utils.mkdir_if_missing(self.save_dir)
        logger.info(f"[ DEBUG | Initialized ReIDTrainer on {self.device} ]")
        logger.info(f"[ DEBUG | Output Directory: {self.save_dir} ]")

    def setup_data(self, 
                   batch_size: int = 128, 
                   height: int = 256, 
                   width: int = 128, 
                   workers: int = 8) -> None:
        """Initialize the ImageDataManager.

        Args:
            batch_size: Batch size for training.
            height: Image height for resizing.
            width: Image width for resizing.
            workers: Number of data loading workers.
        """
        logger.info(f"[ DEBUG | Loading Dataset from: {self.dataset_path} ]")
        
        try:
            self.datamanager = torchreid.data.ImageDataManager(
                root=self.dataset_path,
                sources='SoccerNetReIDDataset', 
                targets='SoccerNetReIDDataset',
                height=height,
                width=width,
                batch_size_train=batch_size,
                batch_size_test=100, 
                transforms=['random_flip', 'random_crop', 'random_erase'], 
                workers=workers,
                use_gpu=(self.device.type == 'cuda')
            )
            
            num_train = self.datamanager.num_train_pids
            num_query = len(self.datamanager.test_loader['SoccerNetReIDDataset']['query'].dataset)
            num_gallery = len(self.datamanager.test_loader['SoccerNetReIDDataset']['gallery'].dataset)
            
            logger.info(f"[ INFO | Dataset Loaded Successfully ]")
            logger.info(f"[ INFO | Identities: {num_train} | Query: {num_query} | Gallery: {num_gallery} ]")

        except Exception as e:
            logger.error(f"[ ERROR | Failed to load dataset: {e} ]")
            raise e

    def setup_model(self, lr: float = 0.0003, max_epoch: int = 10) -> None:
        """Build model, optimizer, and scheduler.

        Args:
            lr: Learning rate.
            max_epoch: Total number of epochs (needed for scheduler).
        """
        if self.datamanager is None:
            raise RuntimeError("[ ERROR | Call setup_data() before setup_model() ]")

        logger.info(f"[ DEBUG | Building Model Arch: {self.model_arch} ]")
        
        try:
            self.model = torchreid.models.build_model(
                name=self.model_arch,
                num_classes=self.datamanager.num_train_pids,
                loss='triplet',
                pretrained=True
            )
            self.model.to(self.device)

            self.optimizer = torchreid.optim.build_optimizer(
                self.model, optim='adam', lr=lr
            )
            
            self.scheduler = torchreid.optim.build_lr_scheduler(
                self.optimizer, lr_scheduler='cosine', max_epoch=max_epoch
            )
            
            self.engine = torchreid.engine.ImageTripletEngine(
                self.datamanager, 
                self.model, 
                optimizer=self.optimizer, 
                scheduler=self.scheduler, 
                label_smooth=True
            )
            logger.info(f"[ DEBUG | Model and Engine initialized ]")

        except Exception as e:
            logger.error(f"[ ERROR | Failed to build model components: {e} ]")
            raise e

    def train(self, max_epoch: int, eval_freq: int = 1) -> None:
        """Execute the training loop.

        Args:
            max_epoch: Total epochs to train.
            eval_freq: Frequency of validation evaluation.
        """
        if self.engine is None:
            raise RuntimeError("[ ERROR | Engine not initialized. Call setup_model() first. ]")

        logger.info(f"[ DEBUG | Starting Training for {max_epoch} epochs ]")
        
        try:
            self.engine.run(
                save_dir=self.save_dir,
                max_epoch=max_epoch,
                eval_freq=eval_freq, 
                print_freq=50,      
                test_only=False,
                visrank=False   
            )
            logger.info(f"[ SUCCESS | Training Completed ]")

        except Exception as e:
            logger.error(f"[ ERROR | Training crashed: {e} ]")
            raise e

    def export(self, export_path: str) -> None:
        """Export the trained model weights to .pt and TorchScript.

        Args:
            export_path: Destination path for the .pt file
        """
        logger.info("[ DEBUG | Starting Export Procedure... ]")
        
        best_ckpt = os.path.join(self.save_dir, 'model-best.pth.tar')
        last_ckpt = os.path.join(self.save_dir, 'model.pth.tar')
        
        target_ckpt = best_ckpt if os.path.exists(best_ckpt) else (last_ckpt if os.path.exists(last_ckpt) else None)

        if not target_ckpt:
            logger.error("[ ERROR | No checkpoint found in output directory. Export failed. ]")
            return

        logger.info(f"[ DEBUG | Loading weights from: {target_ckpt} ]")

        try:
            checkpoint = torch.load(target_ckpt, map_location='cpu')
            weights = checkpoint['state_dict']
            clean_weights = {k.replace('module.', ''): v for k, v in weights.items()}
            
            export_dir = os.path.dirname(export_path)
            os.makedirs(export_dir, exist_ok=True)
            
            torch.save(clean_weights, export_path)
            logger.info(f"[ DEBUG | Weights saved to: {export_path} ]")

            self.model.load_state_dict(clean_weights)
            self.model.eval()
            
            dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
            
            traced_model = torch.jit.trace(self.model, dummy_input)
            ts_path = export_path.replace(".pt", ".torchscript.pt")
            traced_model.save(ts_path)
            logger.info(f"[ DEBUG | TorchScript saved to: {ts_path} ]")

        except Exception as e:
            logger.error(f"[ ERROR | Export failed: {e} ]")


def main():
    DATASET_PATH = "C:/Users/ciroc/Desktop/AV_project/data/datasets/ReidCrop/SoccerNet_Crops_for_REID"
    MODEL_ARCH = 'osnet_x1_0'
    BATCH_SIZE = 128
    MAX_EPOCH = 10
    LR = 0.0003
    DEVICE_ID = 0
    
    RUN_NAME = f'soccernet_{MODEL_ARCH}'
    SAVE_DIR = f'C:/Users/ciroc/Desktop/AV_project/src/training/runs/tracker/{RUN_NAME}'
    EXPORT_PATH = f"models/weights/reid/{MODEL_ARCH}_soccernet.pt"

    trainer = ReIDTrainer(
        dataset_path=DATASET_PATH,
        save_dir=SAVE_DIR,
        model_arch=MODEL_ARCH,
        device_id=DEVICE_ID
    )

    trainer.setup_data(
        batch_size=BATCH_SIZE,
        height=256, 
        width=128, 
        workers=8
    )

    trainer.setup_model(
        lr=LR, 
        max_epoch=MAX_EPOCH
    )

    trainer.train(
        max_epoch=MAX_EPOCH,
        eval_freq=1
    )

    trainer.export(export_path=EXPORT_PATH)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()