"""
app/streamlit_app.py
---------------------
SmartRoad — AI-Powered Street Defect Detection & Reporting System
Streamlit frontend: upload → detect → severity → report → send to municipality.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make src/ importable when running from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    bgr_to_pil,
    get_image_metadata,
    list_sample_images,
    load_image_from_bytes,
    load_image_from_path,
)
from src.detection import DetectionResult, RoadDefectDetector, get_detector
from src.report_generator import OfficialReport, generate_report
from src.severity import SeverityReport, SeverityLevel, compute_severity


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartRoad — Road Defect AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark industrial theme with safety-yellow accent
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .mono {
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Main background */
    .stApp { background-color: #0f1117; color: #e8eaed; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a2f3e; }

    /* Cards */
    .card {
        background: #161b27;
        border: 1px solid #2a2f3e;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-accent {
        border-left: 4px solid #f5c518;
    }

    /* Metric boxes */
    .metric-box {
        background: #1e2436;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .metric-label { font-size: 0.75rem; color: #8b95a8; text-transform: uppercase; letter-spacing: 0.08em; }

    /* Severity badges */
    .badge-critical { background: #7f1d1d; color: #fca5a5; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .badge-high     { background: #7c2d12; color: #fdba74; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .badge-moderate { background: #713f12; color: #fde68a; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
    .badge-low      { background: #14532d; color: #86efac; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }

    /* Report text */
    .report-body { font-size: 0.9rem; line-height: 1.7; white-space: pre-wrap; color: #d1d5db; }

    /* Accent yellow */
    .accent { color: #f5c518; }
    .accent-bold { color: #f5c518; font-weight: 700; }

    /* Progress bar override */
    .stProgress > div > div > div > div { background-color: #f5c518; }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        width: 100%;
    }

    /* Divider */
    hr { border-color: #2a2f3e; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "detection_result": None,
        "severity_report": None,
        "official_report": None,
        "current_image_bgr": None,
        "location_hint": "Unknown road segment",
        "email_sent": False,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_badge(level: SeverityLevel) -> str:
    css_class = f"badge-{level.value.lower()}"
    return f'<span class="{css_class}">{level.value.upper()}</span>'


def _bgr_to_bytes(bgr: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR NumPy image to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()


def _run_pipeline(image_bgr: np.ndarray, location: str) -> None:
    """Run the full detection → severity → report pipeline."""
    st.session_state["processing"] = True

    with st.spinner("Running AI detection…"):
        detector = get_detector()
        result: DetectionResult = detector.detect(image_bgr)
        st.session_state["detection_result"] = result

    with st.spinner("Scoring severity…"):
        sev: SeverityReport = compute_severity(result)
        st.session_state["severity_report"] = sev

    with st.spinner("Generating official report with LLM…"):
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
        st.session_state["official_report"] = report

    st.session_state["processing"] = False
    st.session_state["email_sent"] = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<h2 class="mono accent">🛣️ SmartRoad</h2>'
            '<p style="color:#8b95a8;font-size:0.8rem;margin-top:-0.5rem;">'
            "AI Street Defect Detection v1.0</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Location input
        st.markdown("**📍 Location**")
        loc = st.text_input(
            "Road / intersection description",
            value=st.session_state["location_hint"],
            placeholder="e.g. Main St & 5th Ave, Downtown",
            label_visibility="collapsed",
        )
        st.session_state["location_hint"] = loc

        st.divider()

        # Model settings
        st.markdown("**⚙️ Detection Settings**")
        conf_thresh = st.slider(
            "Confidence threshold", 0.10, 0.90, 0.35, 0.05,
            help="Minimum confidence to show a detection."
        )

        st.divider()

        # LLM provider info
        provider_env = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
        provider_status = "🟢 Groq (Llama 3.1-70B)" if provider_env else "🟡 Mock (template)"
        st.markdown(f"**🤖 LLM Provider**  \n{provider_status}")
        if not provider_env:
            st.info(
                "Set `GROQ_API_KEY` or `LLM_API_KEY` env var to enable AI-generated reports.",
                icon="ℹ️",
            )

        st.divider()
        st.markdown(
            '<p style="color:#8b95a8;font-size:0.72rem;">'
            "SmartRoad © 2025 — for demonstration purposes."
            "</p>",
            unsafe_allow_html=True,
        )

    return conf_thresh  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div style="padding: 1.5rem 0 0.5rem 0;">
          <h1 class="mono" style="font-size:2rem;margin-bottom:0.2rem;">
            <span class="accent">SMART</span>ROAD
          </h1>
          <p style="color:#8b95a8;font-size:0.9rem;margin-top:0;">
            AI-Powered Street Defect Detection &amp; Municipal Reporting System
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

def render_input_section(conf_thresh: float) -> None:
    st.markdown("### 📷 Upload Road Image")

    tab_upload, tab_sample = st.tabs(["Upload your image", "Use a sample"])

    image_bgr: np.ndarray | None = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Drag & drop a road image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            image_bgr = load_image_from_bytes(uploaded.read())
            st.image(bgr_to_pil(image_bgr), caption="Uploaded image", use_container_width=True)

    with tab_sample:
        sample_paths = list_sample_images()
        if not sample_paths:
            st.warning(
                "No sample images found in `data/samples/`.  \n"
                "Run `python scripts/download_samples.py` to download them.",
                icon="⚠️",
            )
        else:
            sample_names = [p.name for p in sample_paths]
            chosen = st.selectbox("Select a sample image", sample_names)
            chosen_path = next(p for p in sample_paths if p.name == chosen)
            image_bgr = load_image_from_path(chosen_path)
            st.image(bgr_to_pil(image_bgr), caption=chosen, use_container_width=True)

    if image_bgr is not None:
        st.session_state["current_image_bgr"] = image_bgr
        meta = get_image_metadata(image_bgr)
        st.caption(
            f"📐 {meta['width']} × {meta['height']} px | "
            f"💾 {meta['size_kb']} KB | "
            f"🎨 {meta['channels']}-channel"
        )

        col_run, col_clear = st.columns([3, 1])
        with col_run:
            if st.button("🔍 Analyse Road Defects", type="primary", use_container_width=True):
                _run_pipeline(image_bgr, st.session_state["location_hint"])
                st.rerun()
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                for k in ["detection_result", "severity_report", "official_report",
                          "current_image_bgr", "email_sent"]:
                    st.session_state[k] = None
                st.rerun()


# ---------------------------------------------------------------------------
# Results section
# ---------------------------------------------------------------------------

def render_results() -> None:
    result: DetectionResult | None = st.session_state.get("detection_result")
    sev: SeverityReport | None = st.session_state.get("severity_report")
    report: OfficialReport | None = st.session_state.get("official_report")

    if result is None or sev is None:
        return

    st.divider()
    st.markdown("### 🔬 Detection Results")

    # --- Annotated image + metrics side by side ---
    col_img, col_stats = st.columns([2, 1], gap="large")

    with col_img:
        annotated_pil = bgr_to_pil(result.annotated_image)
        st.image(annotated_pil, caption="AI-annotated image", use_container_width=True)

    with col_stats:
        # Severity badge
        st.markdown(
            f'<div class="card card-accent">'
            f'<div class="metric-label">Severity Level</div>'
            f'<div class="metric-value" style="color:{sev.colour}">{sev.icon} {sev.level_str}</div>'
            f'<div style="margin-top:0.3rem">{_severity_badge(sev.level)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Score progress
        st.markdown(
            f'<div class="metric-label" style="margin-bottom:4px;">Severity Score</div>',
            unsafe_allow_html=True,
        )
        st.progress(sev.score / 100)
        st.caption(f"{sev.score_int}/100")

        # Quick stats
        metrics = [
            ("Defects Found",    str(sev.defect_count),             ""),
            ("Avg Confidence",   f"{sev.confidence_avg:.0%}",       ""),
            ("Area Affected",    f"{sev.area_coverage_pct:.1f}%",   ""),
            ("Inference Time",   f"{result.inference_time_ms:.0f}ms",""),
        ]
        for label, value, _ in metrics:
            st.markdown(
                f'<div class="metric-box" style="margin-bottom:0.5rem;">'
                f'<div class="metric-value accent-bold">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Defect breakdown ---
    if sev.breakdown:
        st.markdown("#### Detected Defects")
        cols = st.columns(min(len(sev.breakdown), 4))
        for i, (label, count) in enumerate(sev.breakdown.items()):
            with cols[i % len(cols)]:
                st.metric(label.replace("_", " ").title(), count)

    # --- Observations ---
    with st.expander("📋 AI Observations", expanded=True):
        for note in sev.notes:
            st.markdown(f"- {note}")

    # --- Response time ---
    st.info(
        f"⏱️ **Recommended response time:** {sev.recommended_response_time}",
        icon=sev.icon,
    )

    # --- Official report ---
    if report:
        render_report_section(report, sev)


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------

def render_report_section(report: OfficialReport, sev: SeverityReport) -> None:
    st.divider()
    st.markdown("### 📄 Official Municipal Report")

    with st.container():
        st.markdown(
            f'<div class="card">'
            f'<h3 class="mono accent" style="margin:0 0 0.3rem 0;">{report.title}</h3>'
            f'<small style="color:#8b95a8;">'
            f'Report ID: <b>{report.report_id}</b> &nbsp;|&nbsp; '
            f'Generated: {report.generated_at} &nbsp;|&nbsp; '
            f'Model: {report.model}'
            f'</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="card"><div class="report-body">{report.full_text}</div></div>',
            unsafe_allow_html=True,
        )

    # Download + send buttons
    col_dl, col_send = st.columns(2)

    with col_dl:
        report_md = report.as_markdown()
        st.download_button(
            label="📥 Download Report (.md)",
            data=report_md,
            file_name=f"{report.report_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_send:
        if st.button(
            "📧 Send to Municipality",
            type="primary",
            use_container_width=True,
        ):
            _mock_send_email(report, sev)


def _mock_send_email(report: OfficialReport, sev: SeverityReport) -> None:
    """Simulate sending the report to the municipal authority."""
    with st.spinner("Connecting to municipal portal…"):
        time.sleep(1.5)

    # Mock success toast
    st.success(
        f"✅ **Report {report.report_id} sent successfully!**  \n"
        f"Recipient: roads@municipality.gov  \n"
        f"Priority: {sev.level_str}  \n"
        f"Expected response: {sev.recommended_response_time}",
    )
    st.balloons()
    st.session_state["email_sent"] = True

    with st.expander("📬 Email preview (mock)", expanded=False):
        st.code(
            f"TO:      roads@municipality.gov\n"
            f"FROM:    smartroad-system@city.ai\n"
            f"SUBJECT: [SmartRoad] Road Defect Report — {sev.level_str} Severity — {report.report_id}\n\n"
            + report.as_plain_text(),
            language=None,
        )


# ---------------------------------------------------------------------------
# No-results placeholder
# ---------------------------------------------------------------------------

def render_placeholder() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#4b5563;">
          <div style="font-size:3rem;">🛣️</div>
          <div style="font-size:1.1rem;margin-top:0.5rem;">
            Upload a road image to begin defect analysis
          </div>
          <div style="font-size:0.8rem;margin-top:0.3rem;">
            Supported: dashcam photos, aerial images, street-level shots
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conf_thresh = render_sidebar()
    render_header()
    render_input_section(conf_thresh)

    if st.session_state.get("detection_result") is None:
        render_placeholder()
    else:
        render_results()


if __name__ == "__main__":
    main()