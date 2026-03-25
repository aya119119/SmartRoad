"""
src/data_loader.py
------------------
Handles loading and preprocessing of road images and video frames
for the SmartRoad defect detection pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image


# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


def load_image_from_path(image_path: str | Path) -> np.ndarray:
    """
    Load an image from a file path and return it as a NumPy BGR array.

    Args:
        image_path: Path to the image file.

    Returns:
        NumPy array in BGR format (OpenCV convention).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return img


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Load an image from raw bytes (e.g. from Streamlit uploader).

    Args:
        data: Raw image bytes.

    Returns:
        NumPy array in BGR format.
    """
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a NumPy BGR array.

    Args:
        pil_image: A PIL Image object.

    Returns:
        NumPy array in BGR format.
    """
    rgb = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """
    Convert a NumPy BGR array to a PIL Image (RGB).

    Args:
        bgr_image: NumPy array in BGR format.

    Returns:
        PIL Image in RGB format.
    """
    return Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def resize_for_model(
    image: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """
    Resize an image to the target square size while preserving aspect ratio
    with letterboxing (grey padding).

    Args:
        image: Input BGR image.
        target_size: Target width and height (default 640 for YOLOv8).

    Returns:
        Resized and padded BGR image of shape (target_size, target_size, 3).
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create grey canvas and center the resized image
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def video_frame_generator(
    video_path: str | Path,
    frame_skip: int = 5,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, frame_bgr) tuples from a video file.

    Args:
        video_path: Path to the video file.
        frame_skip: Process every N-th frame (default 5 for speed).

    Yields:
        Tuple of (frame_index, BGR frame as NumPy array).

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file is not a supported video format.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def list_sample_images() -> list[Path]:
    """
    Return a sorted list of sample image paths from data/samples/.

    Returns:
        List of Path objects for each sample image.
    """
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_metadata(image: np.ndarray) -> dict:
    """
    Extract basic metadata from a loaded image.

    Args:
        image: NumPy BGR image.

    Returns:
        Dictionary with keys: height, width, channels, size_kb (estimated).
    """
    h, w, c = image.shape
    size_kb = round(image.nbytes / 1024, 1)
    return {"height": h, "width": w, "channels": c, "size_kb": size_kb}