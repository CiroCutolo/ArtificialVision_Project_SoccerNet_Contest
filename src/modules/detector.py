import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

class PlayerDetector:
    """
    Detect players in images using an Ultralytics YOLO model.

    Loads a YOLO model and exposes a detect method to run inference on single frames.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector.

        Args:
            config : dict
                Expected keys:
                - system.device (str): device identifier
                - detection.model_path (str): path to the YOLO weights/file.
                - detection.conf_thres (float): confidence threshold for detections.
                - detection.use_sahi (optional, bool): whether SAHI slicing is enabled (default False).
        """
        self.config = config
        self.device = config["system"]["device"]

        det_conf = config["detection"]
        self.model_path = det_conf["model_path"]
        self.conf_thres = det_conf["conf_thres"]

        print(f"[ Detector | Loading model from: {self.model_path} ]")
        print(f"[ Detector | Config: Conf={self.conf_thres} | Device={self.device} ]")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single image.

        Args:
            frame (ndarray): Image array (H x W x C), BGR or RGB as expected by the model.

        Returns:
            ndarray: Array of detections with shape (N, 6). Each row is [x1, y1, x2, y2, score, class].
                     Returns an empty array with shape (0, 6) when no detections are found.
        """
        detections = []
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_thres,
            augment=True,
            rect=False,
            iou=0.6,
            imgsz=1280,
            classes=[0]
        )[0]

        if len(results.boxes) > 0:
            detections = results.boxes.data.cpu().numpy()
        else:
            detections = []

        return np.array(detections) if len(detections) > 0 else np.empty((0, 6))