"""
app/streamlit_app.py
--------------------
SmartRoad mobile-style interface for video/image upload, AI defect detection,
severity scoring, official report generation, map alerting, and authority
notification.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Map display
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Make src/ importable when running from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    bgr_to_pil,
    get_image_metadata,
    list_sample_images,
    load_frame_from_video_bytes,
    load_image_from_bytes,
    load_image_from_path,
)
from src.database import DatabaseManager
from src.detection import DetectionResult, get_detector
from src.report_generator import OfficialReport, generate_report
from src.severity import SeverityLevel, SeverityReport, compute_severity

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DB = DatabaseManager()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartRoad — Civic Issue Reporting",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .mono {
        font-family: 'IBM Plex Mono', monospace;
    }
    .stApp { background-color: #f8fafc; color: #111827; }
    [data-testid="stSidebar"] { background-color: #111827; color: #f8fafc; }
    .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px; padding: 1.2rem; margin-bottom: 1rem; }
    .card-accent { border-left: 4px solid #ef4444; }
    .metric-box { background: #111827; color: #f8fafc; border-radius: 12px; padding: 1rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; }
    .report-body { font-size: 0.95rem; line-height: 1.75; color: #111827; white-space: pre-wrap; }
    .badge-critical { background: #dc2626; color: #fff; padding: 4px 12px; border-radius: 999px; }
    .badge-high { background: #f97316; color: #fff; padding: 4px 12px; border-radius: 999px; }
    .badge-moderate { background: #facc15; color: #111827; padding: 4px 12px; border-radius: 999px; }
    .badge-low { background: #16a34a; color: #fff; padding: 4px 12px; border-radius: 999px; }
    .stButton > button { border-radius: 12px; font-weight: 600; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "detection_result": None,
        "severity_report": None,
        "official_report": None,
        "current_image_bgr": None,
        "location_hint": "Main St & 5th Ave, Downtown",
        "latitude": "",
        "longitude": "",
        "uploaded_filename": "",
        "notification_sent": False,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _severity_badge(level: SeverityLevel) -> str:
    css_class = f"badge-{level.value.lower()}"
    return f'<span class="{css_class}">{level.value}</span>'


def _bgr_to_bytes(image: np.ndarray, quality: int = 88) -> bytes:
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


def _parse_coordinate(value: str) -> float | None:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _simulate_repair_image(image: np.ndarray, detections: list) -> np.ndarray:
    repaired = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        patch = repaired[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        blurred = cv2.GaussianBlur(patch, (31, 31), 0)
        repaired[y1:y2, x1:x2] = cv2.addWeighted(patch, 0.25, blurred, 0.75, 0)
    cv2.putText(
        repaired,
        "AFTER: Simulated fix",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (34, 197, 94),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return repaired


def _notify_authorities(report: OfficialReport, sev: SeverityReport, location: str, latitude: float | None, longitude: float | None) -> None:
    message = (
        f"[SmartRoad ALERT] {report.report_id} | {sev.level_str} | {location}"
        f" | lat={latitude if latitude is not None else 'N/A'}"
        f" | lon={longitude if longitude is not None else 'N/A'}"
    )
    print(message)
    print(report.as_plain_text())


def _save_report_to_db(report: OfficialReport, sev: SeverityReport, location: str, latitude: float | None, longitude: float | None) -> None:
    issue_type = sev.primary_defect_type if sev.primary_defect_type != "none" else "road_damage"
    DB.insert_report(
        report_id=report.report_id,
        location_description=location,
        latitude=latitude,
        longitude=longitude,
        issue_type=issue_type,
        severity_score=sev.score_int,
        severity_level=sev.level_str,
        report_text=report.as_plain_text(),
    )


def _run_pipeline(image_bgr: np.ndarray, location: str, conf_threshold: float) -> None:
    st.session_state.processing = True
    with st.spinner("Running AI defect detection..."):
        detector = get_detector()
        result: DetectionResult = detector.detect(image_bgr, conf_threshold=conf_threshold)
        st.session_state.detection_result = result

    with st.spinner("Scoring severity..."):
        sev: SeverityReport = compute_severity(result)
        st.session_state.severity_report = sev

    with st.spinner("Generating official report..."):
        report: OfficialReport = generate_report(
            severity_level=sev.level_str,
            defect_count=sev.defect_count,
            breakdown=sev.breakdown,
            score=sev.score,
            area_pct=sev.area_coverage_pct,
            avg_confidence=sev.confidence_avg,
            response_time=sev.recommended_response_time,
            location_hint=location,
            notes=sev.notes,
        )
        st.session_state.official_report = report

    latitude = _parse_coordinate(st.session_state.latitude)
    longitude = _parse_coordinate(st.session_state.longitude)
    _save_report_to_db(report, sev, location, latitude, longitude)
    _notify_authorities(report, sev, location, latitude, longitude)
    st.session_state.notification_sent = True
    st.session_state.processing = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> float:
    with st.sidebar:
        st.markdown(
            '<div style="padding: 1rem 0 0.5rem 0;">'
            '<h2 class="mono" style="margin:0;color:#f8fafc;">🛣️ SmartRoad</h2>'
            '<p style="color:#d1d5db;margin-top:0.25rem;">Civic issue report platform</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**📌 Detection Settings**")
        conf_thresh = st.slider(
            "Confidence threshold",
            0.10,
            0.90,
            0.35,
            0.05,
            help="Minimum AI confidence to keep a detection.",
        )
        st.markdown("---")
        provider_env = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        provider_status = "🟢 AI backend enabled" if provider_env else "🟡 Mock report generation"
        st.markdown(f"**🤖 LLM Provider**  \n{provider_status}")
        if not provider_env:
            st.info("Set OPENAI_API_KEY or GROQ_API_KEY to enable real LLM reports.", icon="ℹ️")
        st.markdown("---")
        open_count = len(DB.get_open_reports())
        st.markdown(f"**📍 Active Alerts**  \n{open_count} open report(s)")
        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.75rem;color:#9ca3af;">SmartRoad MVP — from video to official incident report.</p>',
            unsafe_allow_html=True,
        )
    return conf_thresh


# ---------------------------------------------------------------------------
# Main inputs
# ---------------------------------------------------------------------------

def render_input_section(conf_thresh: float) -> None:
    st.markdown("### 🚨 Report a Civic Issue")
    st.markdown("Upload a short video or image of the issue, then add a location.")

    uploaded = st.file_uploader(
        "Upload photo or video",
        type=[*IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS],
        label_visibility="collapsed",
        help="Supported: JPG, PNG, MP4, MOV, AVI, MKV",
    )

    st.markdown("#### Location")
    location_text = st.text_input(
        "Location description",
        value=st.session_state.location_hint,
        placeholder="Street address, intersection, or landmark",
    )
    st.session_state.location_hint = location_text

    cols = st.columns(2)
    with cols[0]:
        st.session_state.latitude = st.text_input(
            "Latitude",
            value=st.session_state.latitude,
            placeholder="e.g. 37.7749",
        )
    with cols[1]:
        st.session_state.longitude = st.text_input(
            "Longitude",
            value=st.session_state.longitude,
            placeholder="e.g. -122.4194",
        )

    image_bgr = None
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        bytes_data = uploaded.read()
        if suffix in VIDEO_EXTENSIONS:
            try:
                image_bgr = load_frame_from_video_bytes(bytes_data, frame_skip=12)
                st.success("Video frame extracted for AI analysis.")
            except Exception as exc:
                st.error(f"Unable to process video: {exc}")
        else:
            try:
                image_bgr = load_image_from_bytes(bytes_data)
            except Exception as exc:
                st.error(f"Unable to load image: {exc}")

    if image_bgr is not None:
        st.session_state.current_image_bgr = image_bgr
        st.session_state.uploaded_filename = uploaded.name
        st.image(
            bgr_to_pil(image_bgr),
            caption=f"Selected frame from {uploaded.name}",
            use_container_width=True,
        )
        st.caption(
            "If the video contains multiple frames, SmartRoad uses a representative frame for detection."
        )

    if st.button("🔍 Analyse and Report", type="primary", use_container_width=True) and st.session_state.current_image_bgr is not None:
        _run_pipeline(
            st.session_state.current_image_bgr,
            st.session_state.location_hint,
            conf_thresh,
        )
        st.experimental_rerun()

    if st.button("🗑 Clear current report", use_container_width=True):
        for key in [
            "detection_result",
            "severity_report",
            "official_report",
            "current_image_bgr",
            "uploaded_filename",
            "notification_sent",
        ]:
            st.session_state[key] = None
        st.experimental_rerun()


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_results() -> None:
    result: DetectionResult | None = st.session_state.get("detection_result")
    sev: SeverityReport | None = st.session_state.get("severity_report")
    report: OfficialReport | None = st.session_state.get("official_report")

    if result is None or sev is None or report is None:
        return

    st.markdown("### 🔬 Detection Summary")
    before = bgr_to_pil(result.annotated_image)
    after = bgr_to_pil(_simulate_repair_image(result.annotated_image, result.detections))

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(before, caption="Detected issue (AI annotated)", use_container_width=True)
    with col2:
        st.image(after, caption="Simulated fix (after)", use_container_width=True)

    col_a, col_b = st.columns([2, 1], gap="large")
    with col_a:
        if sev.breakdown:
            st.markdown("#### Detected issue types")
            breakdown_text = "\n".join(f"- {label.replace('_', ' ').title()}: {count}" for label, count in sev.breakdown.items())
            st.markdown(breakdown_text)

        with st.expander("📋 AI observations", expanded=True):
            for note in sev.notes:
                st.markdown(f"- {note}")

    with col_b:
        st.markdown(
            f'<div class="card card-accent">'
            f'<div class="metric-label">Severity Level</div>'
            f'<div class="metric-value">{sev.icon} {sev.level_str}</div>'
            f'<div style="margin-top:0.5rem;">{_severity_badge(sev.level)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-box" style="margin-top:1rem;">'
            f'<div class="metric-value">{sev.score_int}/100</div>'
            f'<div class="metric-label">Severity score</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-box" style="margin-top:0.8rem;">'
            f'<div class="metric-value">{sev.recommended_response_time}</div>'
            f'<div class="metric-label">Recommended response</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    render_report_section(report)
    render_map_section()
    render_open_alerts()


# ---------------------------------------------------------------------------
# Report display
# ---------------------------------------------------------------------------

def render_report_section(report: OfficialReport) -> None:
    st.markdown("### 📝 Generated Official Report")
    st.markdown(
        f'<div class="card">'
        f'<h3 class="mono" style="margin:0 0 0.4rem 0;">{report.title}</h3>'
        f'<p style="color:#6b7280; margin:0;">Report ID: <strong>{report.report_id}</strong> • Generated: {report.generated_at}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="card report-body">{report.full_text}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([2, 1])
    with cols[0]:
        report_md = report.as_markdown()
        st.download_button(
            label="📥 Download report",
            data=report_md,
            file_name=f"{report.report_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with cols[1]:
        if st.button("📧 Simulate authority alert", use_container_width=True):
            sev = st.session_state.severity_report
            if sev and report:
                _notify_authorities(report, sev, st.session_state.location_hint, _parse_coordinate(st.session_state.latitude), _parse_coordinate(st.session_state.longitude))
                st.success("Authority notification simulated and printed to console.")


# ---------------------------------------------------------------------------
# Map & open alerts
# ---------------------------------------------------------------------------

def _severity_color(level: str) -> str:
    return {
        "Low": "green",
        "Moderate": "orange",
        "High": "darkorange",
        "Critical": "red",
    }.get(level, "blue")


def render_map_section() -> None:
    st.markdown("---")
    st.markdown("### 📍 Report Map")
    open_reports = DB.get_open_reports()
    markers = [r for r in open_reports if r.latitude is not None and r.longitude is not None]

    if not markers:
        st.info(
            "No geocoded open alerts available yet. Add latitude/longitude to a report to see it on the map.",
            icon="ℹ️",
        )
        return

    if FOLIUM_AVAILABLE:
        center = [markers[0].latitude or 0.0, markers[0].longitude or 0.0]
        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        for row in markers:
            popup_html = (
                f"<strong>{row.issue_type.replace('_', ' ').title()}</strong><br>"
                f"Severity: {row.severity_level} ({row.severity_score}/100)<br>"
                f"Location: {row.location_description}<br>"
                f"Status: {row.status.title()}"
            )
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=9,
                color=_severity_color(row.severity_level),
                fill=True,
                fill_color=_severity_color(row.severity_level),
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(m)
        st_folium(m, width=900, height=480)
    else:
        st.warning("Install folium and streamlit-folium to view the interactive map.")
        st.write({"markers": [dict(r.__dict__) for r in markers]})


def render_open_alerts() -> None:
    st.markdown("---")
    st.markdown("### 🚨 Open Alerts Dashboard")
    open_reports = DB.get_open_reports()
    if not open_reports:
        st.info("No open reports in the system yet.", icon="✅")
        return

    for report in open_reports:
        with st.expander(f"{report.issue_type.replace('_', ' ').title()} — {report.severity_level} ({report.report_id})", expanded=False):
            st.markdown(
                f"**Location:** {report.location_description}<br>",
                f"**Severity:** {report.severity_score}/100 ({report.severity_level})<br>",
                f"**Created:** {report.created_at}<br>",
                f"**Status:** {report.status.title()}",
                unsafe_allow_html=True,
            )
            with cols[0]:
                st.write(report.report_text)
            with cols[1]:
                if st.button("Mark resolved", key=f"resolve_{report.report_id}"):
                    DB.resolve_report(report.report_id)
                    st.success(f"Report {report.report_id} marked resolved.")
                    st.experimental_rerun()


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------

def render_placeholder() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:4rem 1rem;color:#6b7280;">
          <div style="font-size:4rem;">🚦</div>
          <div style="font-size:1.1rem;margin-top:1rem;">Upload a road video or photo to generate a municipal report and alert.</div>
          <div style="font-size:0.9rem;margin-top:0.5rem;">SmartRoad converts visual evidence into an actionable civic issue alert.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conf_thresh = render_sidebar()
    st.title("SmartRoad — Civic Issue Reporting")
    st.markdown("Use the form below to upload evidence, add a location, and create a structured municipal alert.")
    render_input_section(conf_thresh)

    if st.session_state.detection_result is None:
        render_placeholder()
    else:
        render_results()


if __name__ == "__main__":
    main()
