import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import torch
import torchreid

warnings.filterwarnings("ignore")

def setup_logging():
    """Configures the logging format to match project style."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

logger = logging.getLogger(__name__)

def load_and_clean_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    """
    Loads a PyTorch checkpoint and removes DataParallel prefixes ('module.').

    Args:
        ckpt_path: Path to the .pth.tar or .pt file.

    Returns:
        The cleaned state_dict dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the file cannot be loaded.
    """
    if not ckpt_path.exists():
        logger.error(f"[ ERROR | Checkpoint file not found: {ckpt_path} ]")
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"[ INFO | Loading checkpoint: {ckpt_path.name} ]")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"[ ERROR | Failed to load checkpoint: {e} ]")
        raise RuntimeError(e)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        raw_state = checkpoint['state_dict']
        epoch = checkpoint.get('epoch', 'N/A')
        logger.info(f"[ INFO | Detected training checkpoint (Epoch: {epoch}) ]")
    else:
        raw_state = checkpoint
        logger.info("[ INFO | Detected raw state_dict ]")

    clean_state = {}
    for k, v in raw_state.items():
        new_k = k.replace("module.", "")
        clean_state[new_k] = v
    
    return clean_state

def detect_num_classes(state_dict: Dict[str, Any]) -> int:
    """
    Infers the number of classes (identities) from the classifier weights.

    Args:
        state_dict: The model's state dictionary.

    Returns:
        Number of classes detected, or defaults to 1000 if not found.
    """
    if 'classifier.weight' in state_dict:
        num_classes = state_dict['classifier.weight'].shape[0]
        logger.info(f"[ DEBUG | Detected {num_classes} identities from classifier weights ]")
        return num_classes
    
    logger.warning("[ WARN | Could not detect num_classes. Defaulting to 1000. ]")
    return 1000

def export_torchscript(model: torch.nn.Module, output_path: Path) -> None:
    """
    Traces the model and saves it as TorchScript for C++/Deployment.

    Args:
        model: The PyTorch model (in eval mode).
        output_path: Destination path for the .torchscript.pt file.
    """
    # Dummy input for tracing (Batch, Channels, Height, Width)
    # Standard ReID size is usually 256x128
    dummy_input = torch.randn(1, 3, 256, 128)
    
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(output_path)
        logger.info(f"[ SUCCESS | TorchScript exported: {output_path} ]")
    except Exception as e:
        logger.error(f"[ ERROR | TorchScript export failed: {e} ]")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Export TorchReID Model for Inference.")
    parser.add_argument("--ckpt", type=Path, default="C:/Users/ciroc/Desktop/AV_project/src/training/runs/tracker/soccernet_finetune_sportsmot_256x128/model/model.pth.tar-60", help="Path to input checkpoint.")
    parser.add_argument("--arch", type=str, default="osnet_ain_x1_0", help="Model architecture name.")
    parser.add_argument("--out-dir", type=Path, default="C:/Users/ciroc/Desktop/AV_project/data/models/weights/tracker", help="Directory for output files.")
    
    args = parser.parse_args()
    
    try:
        state_dict = load_and_clean_checkpoint(args.ckpt)
    except Exception:
        sys.exit(1)
        
    num_classes = detect_num_classes(state_dict)

    logger.info(f"[ INFO | Building architecture: {args.arch} ]")
    try:
        model = torchreid.models.build_model(
            name=args.arch,
            num_classes=num_classes,
            pretrained=False 
        )
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        logger.error(f"[ ERROR | Failed to build/load model: {e} ]")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    pt_path = args.out_dir / f"{args.arch}_soccernet_v2.pt"
    torch.save(state_dict, pt_path)
    logger.info(f"[ SUCCESS | Clean weights saved: {pt_path} ]")

    ts_path = args.out_dir / f"{args.arch}_soccernet.torchscript.pt"
    export_torchscript(model, ts_path)

if __name__ == "__main__":
    main()