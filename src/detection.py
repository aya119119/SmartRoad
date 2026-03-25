"""
src/detection.py
----------------
Core defect detection module for SmartRoad.

Loads a fine-tuned YOLOv8 pothole/defect detection model from Hugging Face
and runs inference on road images.

Model: keremberke/yolov8n-pothole-segmentation (public, well-maintained)
Fallback: cazzz307/Pothole-Finetuned-YoloV8
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation when ultralytics not installed
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("[detection] ultralytics not installed — running in MOCK mode.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary model: well-known public pothole model on Hugging Face Hub
PRIMARY_MODEL_ID = "keremberke/yolov8n-pothole-segmentation"

# Fallback model
FALLBACK_MODEL_ID = "cazzz307/Pothole-Finetuned-YoloV8"

# Local cache directory for downloaded models
MODEL_CACHE_DIR = Path(__file__).parent.parent / "models"

# Detection confidence threshold
DEFAULT_CONF_THRESHOLD = 0.35

# Class label mapping (depends on which model loads)
DEFAULT_LABEL_MAP: dict[int, str] = {
    0: "pothole",
    1: "crack",
    2: "road_damage",
}

# Colours for drawing bounding boxes (BGR)
BOX_COLOURS: dict[str, tuple[int, int, int]] = {
    "pothole":    (0,   60,  220),   # red-ish
    "crack":      (0,  165,  255),   # orange
    "road_damage":(0,  200,  100),   # green
    "defect":     (180,  50, 220),   # purple fallback
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected defect in an image."""

    label: str                      # e.g. "pothole"
    confidence: float               # 0.0 – 1.0
    bbox: tuple[int, int, int, int] # x1, y1, x2, y2 (pixel coords)
    area_px: int = 0                # bounding-box area in pixels
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # segmentation mask

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if self.area_px == 0:
            self.area_px = max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class DetectionResult:
    """Aggregated detection results for one image."""

    detections: list[Detection]
    annotated_image: np.ndarray        # BGR image with boxes drawn
    inference_time_ms: float
    model_id: str
    image_shape: tuple[int, int]       # (height, width)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.detections]

    @property
    def highest_confidence(self) -> float:
        return max((d.confidence for d in self.detections), default=0.0)

    @property
    def total_defect_area_px(self) -> int:
        return sum(d.area_px for d in self.detections)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class RoadDefectDetector:
    """
    Wraps a YOLOv8 model for road defect detection.

    Usage:
        detector = RoadDefectDetector()
        result = detector.detect(bgr_image)
    """

    def __init__(
        self,
        model_id: str = PRIMARY_MODEL_ID,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        """
        Initialise and load the model.

        Args:
            model_id: Hugging Face model ID or local path.
            conf_threshold: Minimum confidence to keep a detection.
            device: "cpu" or "cuda".
        """
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.device = device
        self._model: Optional[object] = None
        self._mock_mode = not ULTRALYTICS_AVAILABLE

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download / load the YOLOv8 model from Hugging Face."""
        if self._mock_mode:
            print("[detector] Running in MOCK mode (no ultralytics).")
            return

        print(f"[detector] Loading model: {self.model_id} …")
        try:
            # ultralytics supports 'hf://owner/repo' format directly
            hf_path = f"hf://{self.model_id}"
            self._model = YOLO(hf_path)
            print(f"[detector] ✓ Model loaded from Hugging Face: {self.model_id}")
        except Exception as exc:
            print(f"[detector] Primary model failed ({exc}), trying fallback …")
            try:
                hf_fallback = f"hf://{FALLBACK_MODEL_ID}"
                self._model = YOLO(hf_fallback)
                self.model_id = FALLBACK_MODEL_ID
                print(f"[detector] ✓ Fallback model loaded: {FALLBACK_MODEL_ID}")
            except Exception as exc2:
                print(f"[detector] Both models failed ({exc2}). Switching to MOCK mode.")
                self._mock_mode = True

    # ------------------------------------------------------------------
    # Main detection method
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run defect detection on a BGR image.

        Args:
            image: NumPy array in BGR format.
            conf_threshold: Override instance confidence threshold.

        Returns:
            DetectionResult with all detections and annotated image.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        h, w = image.shape[:2]
        t_start = time.perf_counter()

        if self._mock_mode:
            detections = self._mock_detect(image, conf)
        else:
            detections = self._real_detect(image, conf)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        annotated = self._draw_detections(image.copy(), detections)

        return DetectionResult(
            detections=detections,
            annotated_image=annotated,
            inference_time_ms=round(elapsed_ms, 1),
            model_id=self.model_id,
            image_shape=(h, w),
        )

    # ------------------------------------------------------------------
    # Internal: real inference
    # ------------------------------------------------------------------

    def _real_detect(
        self,
        image: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """Run actual YOLOv8 inference."""
        # Convert BGR → RGB for ultralytics
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            source=rgb,
            conf=conf,
            verbose=False,
            device=self.device,
        )

        detections: list[Detection] = []
        for result in results:
            names: dict[int, str] = result.names  # class index → name

            # Bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = names.get(cls_id, f"class_{cls_id}")
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            label=label,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                        )
                    )

            # Segmentation masks (if model supports them)
            if result.masks is not None:
                for i, mask_data in enumerate(result.masks.data):
                    if i < len(detections):
                        mask_np = mask_data.cpu().numpy().astype(np.uint8) * 255
                        detections[i].mask = mask_np

        return detections

    # ------------------------------------------------------------------
    # Internal: mock detection for demo / testing
    # ------------------------------------------------------------------

    def _mock_detect(
        self,
        image: np.ndarray,
        conf: float,
    ) -> list[Detection]:
        """
        Generate plausible fake detections based on image brightness/texture.
        Used when ultralytics is unavailable.
        """
        h, w = image.shape[:2]
        rng = np.random.default_rng(seed=int(image.mean()))  # deterministic per image

        num_defects = rng.integers(1, 4)
        mock_labels = ["pothole", "crack", "road_damage"]
        detections: list[Detection] = []

        for _ in range(num_defects):
            label = rng.choice(mock_labels)
            cx = int(rng.uniform(0.15, 0.85) * w)
            cy = int(rng.uniform(0.15, 0.85) * h)
            bw = int(rng.uniform(0.05, 0.20) * w)
            bh = int(rng.uniform(0.05, 0.18) * h)
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)
            mock_conf = float(rng.uniform(conf, 0.95))
            detections.append(
                Detection(
                    label=label,
                    confidence=round(mock_conf, 3),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.

        Args:
            image: BGR image to annotate (will be modified in place).
            detections: List of Detection objects.

        Returns:
            Annotated BGR image.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = BOX_COLOURS.get(det.label, BOX_COLOURS["defect"])
            label_text = f"{det.label} {det.confidence:.0%}"

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness=2)

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(y1 - 5, th + baseline)
            cv2.rectangle(
                image,
                (x1, label_y - th - baseline),
                (x1 + tw + 4, label_y + baseline),
                colour,
                cv2.FILLED,
            )

            # Label text (white on colour background)
            cv2.putText(
                image,
                label_text,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Optional: draw segmentation mask overlay
            if det.mask is not None:
                try:
                    mask_resized = cv2.resize(
                        det.mask, (image.shape[1], image.shape[0])
                    )
                    coloured_mask = np.zeros_like(image)
                    coloured_mask[mask_resized > 127] = colour
                    image = cv2.addWeighted(image, 1.0, coloured_mask, 0.4, 0)
                except Exception:
                    pass  # silently skip mask overlay errors

        return image


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_detector_instance: Optional[RoadDefectDetector] = None


def get_detector(
    model_id: str = PRIMARY_MODEL_ID,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> RoadDefectDetector:
    """
    Return a shared RoadDefectDetector instance (loaded once per process).

    Args:
        model_id: Hugging Face model ID (only used on first call).
        conf_threshold: Detection confidence threshold.

    Returns:
        RoadDefectDetector instance.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RoadDefectDetector(
            model_id=model_id,
            conf_threshold=conf_threshold,
        )
    return _detector_instance