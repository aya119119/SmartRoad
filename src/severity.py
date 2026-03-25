"""
src/severity.py
---------------
Scores the severity of detected road defects and assigns a priority level
for municipal response.

Severity is computed from:
  - Number of detections
  - Types of defects (pothole > crack > general damage)
  - Individual detection confidence scores
  - Total defect area as a proportion of the image
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.detection import DetectionResult


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    """Road defect severity classification."""
    LOW      = "Low"
    MODERATE = "Moderate"
    HIGH     = "High"
    CRITICAL = "Critical"


# Weights per defect type (higher = more dangerous)
DEFECT_WEIGHTS: dict[str, float] = {
    "pothole":     1.0,
    "crack":       0.55,
    "road_damage": 0.70,
    "defect":      0.60,   # generic fallback label
}

# Severity score thresholds (0 – 100 scale)
SEVERITY_THRESHOLDS: list[tuple[float, SeverityLevel]] = [
    (75.0, SeverityLevel.CRITICAL),
    (50.0, SeverityLevel.HIGH),
    (25.0, SeverityLevel.MODERATE),
    (0.0,  SeverityLevel.LOW),
]

# Recommended municipal response times per severity level
RESPONSE_TIMES: dict[SeverityLevel, str] = {
    SeverityLevel.CRITICAL: "24–48 hours",
    SeverityLevel.HIGH:     "3–5 business days",
    SeverityLevel.MODERATE: "1–2 weeks",
    SeverityLevel.LOW:      "Next scheduled maintenance cycle",
}

# Urgency colours (hex) for UI display
SEVERITY_COLOURS: dict[SeverityLevel, str] = {
    SeverityLevel.CRITICAL: "#DC2626",   # red
    SeverityLevel.HIGH:     "#EA580C",   # orange
    SeverityLevel.MODERATE: "#CA8A04",   # amber
    SeverityLevel.LOW:      "#16A34A",   # green
}

# Emoji indicators
SEVERITY_ICONS: dict[SeverityLevel, str] = {
    SeverityLevel.CRITICAL: "🔴",
    SeverityLevel.HIGH:     "🟠",
    SeverityLevel.MODERATE: "🟡",
    SeverityLevel.LOW:      "🟢",
}


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class SeverityReport:
    """Complete severity assessment for a single road image analysis."""

    score: float                          # 0 – 100 numeric score
    level: SeverityLevel
    defect_count: int
    primary_defect_type: str              # most common defect label
    confidence_avg: float                 # average detection confidence
    area_coverage_pct: float             # % of image covered by defects
    recommended_response_time: str
    colour: str                           # hex for UI
    icon: str                             # emoji
    breakdown: dict[str, int]            # {label: count}
    notes: list[str]                     # human-readable observations

    @property
    def level_str(self) -> str:
        return self.level.value

    @property
    def score_int(self) -> int:
        return round(self.score)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def compute_severity(result: "DetectionResult") -> SeverityReport:
    """
    Compute a severity score and level from a DetectionResult.

    Args:
        result: Output from RoadDefectDetector.detect().

    Returns:
        SeverityReport with all assessment details.
    """
    detections = result.detections
    img_h, img_w = result.image_shape
    total_pixels = img_h * img_w

    # --- Handle zero detections ---
    if not detections:
        return SeverityReport(
            score=0.0,
            level=SeverityLevel.LOW,
            defect_count=0,
            primary_defect_type="none",
            confidence_avg=0.0,
            area_coverage_pct=0.0,
            recommended_response_time=RESPONSE_TIMES[SeverityLevel.LOW],
            colour=SEVERITY_COLOURS[SeverityLevel.LOW],
            icon=SEVERITY_ICONS[SeverityLevel.LOW],
            breakdown={},
            notes=["No road defects detected in this image."],
        )

    # --- Basic metrics ---
    labels = [d.label for d in detections]
    confidences = [d.confidence for d in detections]
    areas = [d.area_px for d in detections]

    # Count per label
    breakdown: dict[str, int] = {}
    for label in labels:
        breakdown[label] = breakdown.get(label, 0) + 1

    primary_defect = max(breakdown, key=lambda k: breakdown[k])
    avg_confidence = sum(confidences) / len(confidences)
    total_defect_area = sum(areas)
    area_coverage_pct = min(100.0, (total_defect_area / total_pixels) * 100)

    # --- Composite score (0 – 100) ---
    # Component 1: Type weight (0–35 points)
    type_score = sum(
        DEFECT_WEIGHTS.get(d.label, 0.5) * d.confidence for d in detections
    )
    type_component = min(35.0, type_score * 15.0)

    # Component 2: Count factor (0–25 points)
    count_component = min(25.0, len(detections) * 6.5)

    # Component 3: Confidence (0–20 points)
    conf_component = avg_confidence * 20.0

    # Component 4: Area coverage (0–20 points)
    area_component = min(20.0, area_coverage_pct * 2.5)

    raw_score = type_component + count_component + conf_component + area_component
    score = min(100.0, round(raw_score, 1))

    # --- Map score → level ---
    level = SeverityLevel.LOW
    for threshold, lvl in SEVERITY_THRESHOLDS:
        if score >= threshold:
            level = lvl
            break

    # --- Human-readable notes ---
    notes = _generate_notes(
        detections=detections,
        breakdown=breakdown,
        score=score,
        level=level,
        area_coverage_pct=area_coverage_pct,
    )

    return SeverityReport(
        score=score,
        level=level,
        defect_count=len(detections),
        primary_defect_type=primary_defect,
        confidence_avg=round(avg_confidence, 3),
        area_coverage_pct=round(area_coverage_pct, 2),
        recommended_response_time=RESPONSE_TIMES[level],
        colour=SEVERITY_COLOURS[level],
        icon=SEVERITY_ICONS[level],
        breakdown=breakdown,
        notes=notes,
    )


def _generate_notes(
    detections: list,
    breakdown: dict[str, int],
    score: float,
    level: SeverityLevel,
    area_coverage_pct: float,
) -> list[str]:
    """Generate concise human-readable observation strings."""
    notes: list[str] = []

    total = len(detections)
    notes.append(
        f"{total} defect{'s' if total != 1 else ''} detected "
        f"across {len(breakdown)} category{'s' if len(breakdown) != 1 else ''}."
    )

    for label, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        notes.append(f"  • {count} {label.replace('_', ' ')}(s) identified.")

    if area_coverage_pct > 10:
        notes.append(
            f"Defects cover approximately {area_coverage_pct:.1f}% of the visible road surface."
        )

    if level == SeverityLevel.CRITICAL:
        notes.append("⚠ Immediate safety hazard — road may be unsafe for vehicles.")
    elif level == SeverityLevel.HIGH:
        notes.append("Road condition requires prompt attention to prevent further deterioration.")
    elif level == SeverityLevel.MODERATE:
        notes.append("Defects are present but not immediately dangerous; monitor and schedule repair.")
    else:
        notes.append("Minor surface irregularities noted; routine maintenance sufficient.")

    return notes