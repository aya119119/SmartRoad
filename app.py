"""
SmartRoad - Road Defect Detection

Main Streamlit interface with image/video upload, interactive map location
selection, YOLOv8 defect detection, severity scoring, and results display.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import folium
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# Make src/ importable when running from project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    bgr_to_pil,
    load_frame_from_video_bytes,
    load_image_from_bytes,
)
from src.detection import DetectionResult, get_detector
from src.report_generator import OfficialReport, generate_report
from src.severity import SeverityLevel, SeverityReport, compute_severity

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MAP_CENTER = [34.0181, -5.0078]
DEFAULT_MAP_ZOOM = 13


def init_session_state() -> None:
    """Initialize session state values used by the app."""
    defaults: dict[str, Any] = {
        "current_image_bgr": None,
        "uploaded_filename": "",
        "location_description": "",
        "selected_lat": None,
        "selected_lon": None,
        "detection_result": None,
        "severity_report": None,
        "official_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_page_config() -> None:
    st.set_page_config(
        page_title="SmartRoad - Road Defect Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_style() -> None:
    st.markdown(
        """
        <style>
        body, .stApp, .css-1y4p8pa { background-color: #ffffff; color: #111111; }
        .stButton>button, .stDownloadButton>button { border-radius: 10px; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div>div {
            border-radius: 10px; border: 1px solid #d1d5db; background-color: #fafafa;
        }
        .stSidebar { background-color: #f8f8f8; }
        .section-box { border: 1px solid #e5e7eb; border-radius: 16px; padding: 1.25rem; margin-bottom: 1.25rem; }
        .section-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.75rem; }
        .small-meta { color: #6b7280; font-size: 0.9rem; }
        .result-text { color: #111827; line-height: 1.7; }
        .metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 14px; padding: 1rem; }
        .metric-value { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
        .metric-label { color: #6b7280; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_uploaded_media(uploaded: Any) -> tuple[np.ndarray | None, str]:
    """Load an image or video from uploaded bytes and return a representative image."""
    if uploaded is None:
        return None, ""

    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.read()
    if suffix in VIDEO_EXTENSIONS:
        frame = load_frame_from_video_bytes(data)
        return frame, uploaded.name

    if suffix in IMAGE_EXTENSIONS:
        image = load_image_from_bytes(data)
        return image, uploaded.name

    return None, ""


def build_map() -> dict[str, Any] | None:
    """Render an interactive map and capture user clicks into session state."""
    center = DEFAULT_MAP_CENTER
    folium_map = folium.Map(
        location=center,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if st.session_state.selected_lat is not None and st.session_state.selected_lon is not None:
        folium.CircleMarker(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            radius=8,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.9,
            popup="Selected location",
        ).add_to(folium_map)

    map_data = st_folium(folium_map, width=800, height=460)
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        st.session_state.selected_lat = click.get("lat")
        st.session_state.selected_lon = click.get("lng")

    return map_data


def run_detection(confidence_threshold: float) -> None:
    """Run the YOLOv8 defect detection pipeline and save results in session state."""
    image_bgr = st.session_state.current_image_bgr
    if image_bgr is None:
        return

    detector = get_detector()
    detection_result: DetectionResult = detector.detect(
        image_bgr, conf_threshold=confidence_threshold
    )
    severity_report: SeverityReport = compute_severity(detection_result)
    location_label = st.session_state.location_description or "Unknown location"

    official_report: OfficialReport = generate_report(
        severity_level=severity_report.level.value,
        defect_count=severity_report.defect_count,
        breakdown=severity_report.breakdown,
        score=severity_report.score,
        area_pct=severity_report.area_coverage_pct,
        avg_confidence=severity_report.confidence_avg,
        response_time=severity_report.recommended_response_time,
        location_hint=location_label,
        notes=severity_report.notes,
    )

    st.session_state.detection_result = detection_result
    st.session_state.severity_report = severity_report
    st.session_state.official_report = official_report


def render_results() -> None:
    """Display annotated image, severity metrics, and generated report."""
    result = st.session_state.detection_result
    severity = st.session_state.severity_report
    report = st.session_state.official_report
    if result is None or severity is None or report is None:
        return

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Results</div>", unsafe_allow_html=True)
    cols = st.columns([2, 1], gap="large")
    with cols[0]:
        st.image(bgr_to_pil(result.annotated_image), use_column_width=True)
    with cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{severity.score:.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Severity score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-card' style='margin-top:1rem;'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{severity.level.value}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Severity level</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-card' style='margin-top:1rem;'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{severity.defect_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Detected defects</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Generated report</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-text'>{report.full_text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    set_page_config()
    init_session_state()
    render_style()

    st.title("SmartRoad - Road Defect Detection")

    with st.sidebar:
        st.header("Configuration")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
        )
        st.markdown("---")
        st.write("Select an image or short video, set the confidence threshold, then choose a location on the map.")

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>1. Upload image or video</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload photo or video",
        type=[*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image_bgr, filename = load_uploaded_media(uploaded_file)
        if image_bgr is not None:
            st.session_state.current_image_bgr = image_bgr
            st.session_state.uploaded_filename = filename
            st.image(bgr_to_pil(image_bgr), use_column_width=True)
        else:
            st.error("Unsupported file type. Please upload a supported image or video.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2. Interactive map location selection</div>", unsafe_allow_html=True)

    st.session_state.location_description = st.text_input(
        "Optional location description",
        value=st.session_state.location_description,
        placeholder="Street address, intersection, or landmark",
    )

    map_container = st.container()
    with map_container:
        map_data = build_map()

    if st.button("Reset map selection"):
        st.session_state.selected_lat = None
        st.session_state.selected_lon = None
        st.experimental_rerun()

    if st.session_state.selected_lat is not None and st.session_state.selected_lon is not None:
        st.markdown(
            f"Selected location: {st.session_state.selected_lat:.6f}, {st.session_state.selected_lon:.6f}"
        )
    else:
        st.markdown("Selected location: none")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3. Run detection</div>", unsafe_allow_html=True)

    run_button = st.button("Run detection")
    if run_button:
        if st.session_state.current_image_bgr is None:
            st.error("Please upload an image or video before running detection.")
        elif st.session_state.selected_lat is None or st.session_state.selected_lon is None:
            st.error("Please select a location on the map before running detection.")
        else:
            run_detection(confidence_threshold)
            st.success("Detection complete. Scroll down for results.")

    st.markdown("</div>", unsafe_allow_html=True)

    render_results()


if __name__ == "__main__":
    main()
