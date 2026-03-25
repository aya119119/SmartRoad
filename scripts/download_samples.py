"""
scripts/download_samples.py
----------------------------
Downloads a small set of royalty-free road/pothole images from
Wikimedia Commons and saves them to data/samples/.

Run:
    python scripts/download_samples.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from repo root or scripts/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

# Royalty-free / CC-licensed road & pothole images from Wikimedia Commons
SAMPLE_IMAGES: list[dict] = [
    {
        "filename": "pothole_road_1.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/"
            "Pothole_in_road.jpg/640px-Pothole_in_road.jpg"
        ),
        "description": "Large pothole in asphalt road",
    },
    {
        "filename": "pothole_road_2.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/"
            "Pothole_in_Chicago.jpg/640px-Pothole_in_Chicago.jpg"
        ),
        "description": "Urban pothole with water",
    },
    {
        "filename": "road_crack_1.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/"
            "Cracked_road_surface.jpg/640px-Cracked_road_surface.jpg"
        ),
        "description": "Surface cracks in road pavement",
    },
    {
        "filename": "damaged_road_1.jpg",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/"
            "Road_surface_damage.jpg/640px-Road_surface_damage.jpg"
        ),
        "description": "General road surface damage",
    },
]

# Fallback URLs using Unsplash (free, no API key needed for small images)
FALLBACK_IMAGES: list[dict] = [
    {
        "filename": "sample_road_1.jpg",
        "url": "https://images.unsplash.com/photo-1515162816999-a0c47dc192f7?w=640&q=80",
        "description": "Road with visible damage",
    },
    {
        "filename": "sample_road_2.jpg",
        "url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=640&q=80",
        "description": "Cracked road surface",
    },
    {
        "filename": "sample_road_3.jpg",
        "url": "https://images.unsplash.com/photo-1601758125946-6ec2ef64daf8?w=640&q=80",
        "description": "Urban road pothole",
    },
    {
        "filename": "sample_road_4.jpg",
        "url": "https://images.unsplash.com/photo-1503387837-b154d5074bd2?w=640&q=80",
        "description": "Highway road surface",
    },
]


def download_image(url: str, dest: Path, timeout: int = 15) -> bool:
    """
    Download a single image from a URL.

    Args:
        url: Source URL.
        dest: Destination file path.
        timeout: Request timeout in seconds.

    Returns:
        True if successful, False otherwise.
    """
    try:
        headers = {"User-Agent": "SmartRoad-SampleDownloader/1.0"}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()

        # Verify it looks like an image
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and len(response.content) < 1000:
            print(f"  ⚠ Skipped (not an image): {url}")
            return False

        dest.write_bytes(response.content)
        size_kb = len(response.content) // 1024
        print(f"  ✓ Saved {dest.name} ({size_kb} KB)")
        return True

    except Exception as exc:
        print(f"  ✗ Failed ({exc}): {url}")
        return False


def main() -> None:
    """Download sample images to data/samples/."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to: {SAMPLES_DIR.resolve()}\n")

    downloaded = 0

    # Try primary sources first
    print("Attempting primary image sources (Wikimedia Commons)...")
    for item in SAMPLE_IMAGES:
        dest = SAMPLES_DIR / item["filename"]
        if dest.exists():
            print(f"  ⏭ Already exists: {item['filename']}")
            downloaded += 1
            continue
        if download_image(item["url"], dest):
            downloaded += 1

    # If primary sources failed, use fallbacks
    if downloaded < 2:
        print("\nTrying fallback image sources (Unsplash)...")
        for item in FALLBACK_IMAGES:
            dest = SAMPLES_DIR / item["filename"]
            if dest.exists():
                print(f"  ⏭ Already exists: {item['filename']}")
                downloaded += 1
                continue
            if download_image(item["url"], dest):
                downloaded += 1
            if downloaded >= 4:
                break

    print(f"\n{'='*50}")
    if downloaded > 0:
        print(f"✅ {downloaded} sample image(s) ready in data/samples/")
        print("\nListing files:")
        for f in sorted(SAMPLES_DIR.iterdir()):
            size_kb = f.stat().st_size // 1024
            print(f"  • {f.name} ({size_kb} KB)")
    else:
        print("❌ Could not download any images.")
        print(
            "\nTo add samples manually, place .jpg/.png files in:\n"
            f"  {SAMPLES_DIR.resolve()}"
        )
    print(
        "\n💡 Tip: You can also use your own road/pothole images."
        "\n   The model works best with clear dashcam or overhead shots."
    )


if __name__ == "__main__":
    main()