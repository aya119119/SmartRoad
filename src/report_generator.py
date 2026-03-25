"""
src/report_generator.py
------------------------
Generates official-style municipal road maintenance reports using an LLM.

Supported providers (set via environment variables):
  - Groq  (GROQ_API_KEY)     — uses llama-3.1-70b-versatile
  - OpenAI (OPENAI_API_KEY)  — uses gpt-4o-mini
  - Mock  (no key needed)    — returns a template-based report

The environment variable LLM_PROVIDER controls which backend is used
("groq" | "openai" | "mock"). Defaults to auto-detection.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Lazy imports for optional LLM clients
_groq_client = None
_openai_client = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL   = "llama-3.1-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"

MAX_REPORT_TOKENS = 900


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class OfficialReport:
    """Structured official road defect report."""

    # Core content
    title: str
    executive_summary: str
    full_text: str              # complete LLM-generated report body
    generated_at: str           # ISO-format timestamp
    report_id: str              # unique reference number

    # Metadata
    provider: str               # "groq" | "openai" | "mock"
    model: str
    location_hint: str
    severity_level: str
    defect_count: int

    def as_markdown(self) -> str:
        """Return the report formatted as Markdown."""
        return (
            f"# {self.title}\n\n"
            f"**Report ID:** {self.report_id}  \n"
            f"**Generated:** {self.generated_at}  \n"
            f"**Location:** {self.location_hint}  \n"
            f"**Severity:** {self.severity_level}  \n\n"
            f"---\n\n"
            f"{self.full_text}"
        )

    def as_plain_text(self) -> str:
        """Return the report as plain text for email body."""
        return (
            f"{self.title}\n"
            f"{'='*len(self.title)}\n\n"
            f"Report ID : {self.report_id}\n"
            f"Generated  : {self.generated_at}\n"
            f"Location   : {self.location_hint}\n"
            f"Severity   : {self.severity_level}\n\n"
            f"{'-'*60}\n\n"
            f"{self.full_text}"
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    severity_level: str,
    defect_count: int,
    breakdown: dict[str, int],
    score: float,
    area_pct: float,
    avg_confidence: float,
    response_time: str,
    location_hint: str,
    notes: list[str],
) -> str:
    """
    Build the LLM prompt that requests a formal municipal report.

    Returns a detailed system + user prompt string.
    """
    breakdown_str = ", ".join(
        f"{v} {k.replace('_', ' ')}(s)" for k, v in breakdown.items()
    )
    notes_str = "\n".join(f"- {n}" for n in notes)

    prompt = textwrap.dedent(f"""
        You are a professional civil-engineering report writer for a municipal
        road maintenance authority. Write a formal, concise official report based
        on the following AI-detected road defect data.

        === DETECTION DATA ===
        Location description : {location_hint}
        Severity level       : {severity_level}
        Severity score       : {score:.0f}/100
        Total defects found  : {defect_count}
        Defect breakdown     : {breakdown_str}
        Avg detection conf.  : {avg_confidence:.0%}
        Road area affected   : {area_pct:.1f}%
        Recommended response : {response_time}

        Observations from AI analysis:
        {notes_str}

        === INSTRUCTIONS ===
        Write a complete official road defect report with these sections:
        1. EXECUTIVE SUMMARY     (2–3 sentences, plain language)
        2. DEFECT DESCRIPTION    (technical details of what was found)
        3. RISK ASSESSMENT       (safety risks to road users)
        4. RECOMMENDED ACTIONS   (specific repair steps, ordered by priority)
        5. TIMELINE & PRIORITY   (based on severity level)
        6. CLOSING NOTE          (escalation guidance if needed)

        Tone: formal, factual, professional government document.
        Do NOT use bullet points — use numbered lists or prose paragraphs only.
        Do NOT invent data not present above.
        Length: 300–450 words total.
    """).strip()

    return prompt


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    """Detect which LLM provider to use based on environment variables."""
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in ("groq", "openai", "mock"):
        return explicit
    if os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def _call_groq(prompt: str) -> tuple[str, str]:
    """Call Groq API and return (text, model_name)."""
    global _groq_client
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY or LLM_API_KEY environment variable not set.")

    if _groq_client is None:
        _groq_client = Groq(api_key=api_key)

    response = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_REPORT_TOKENS,
        temperature=0.4,
    )
    text = response.choices[0].message.content.strip()
    return text, GROQ_MODEL


def _call_openai(prompt: str) -> tuple[str, str]:
    """Call OpenAI API and return (text, model_name)."""
    global _openai_client
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or LLM_API_KEY environment variable not set.")

    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)

    response = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_REPORT_TOKENS,
        temperature=0.4,
    )
    text = response.choices[0].message.content.strip()
    return text, OPENAI_MODEL


def _mock_report(
    severity_level: str,
    defect_count: int,
    breakdown: dict[str, int],
    score: float,
    response_time: str,
    location_hint: str,
) -> tuple[str, str]:
    """
    Return a high-quality template report when no LLM API key is available.
    """
    breakdown_str = ", ".join(
        f"{v} {k.replace('_', ' ')}(s)" for k, v in (breakdown or {"pothole": 1}).items()
    )
    text = textwrap.dedent(f"""
        EXECUTIVE SUMMARY

        An automated inspection of the road segment identified {defect_count}
        defect(s) ({breakdown_str}) with an overall severity score of {score:.0f}/100
        ({severity_level}). Immediate follow-up is recommended within {response_time}.

        DEFECT DESCRIPTION

        The AI-assisted road surface analysis detected the following defects at
        the reported location ({location_hint}): {breakdown_str}. These defects
        were identified with high model confidence and represent measurable
        deterioration of the road surface requiring professional assessment.

        RISK ASSESSMENT

        The identified defects present a risk to road users proportional to their
        severity classification ({severity_level}). Potholes and deep cracks pose
        a direct hazard to vehicle tyres, suspensions, and cyclist safety. Left
        unaddressed, the defects are likely to expand due to weather and traffic
        loading, increasing both repair costs and liability exposure.

        RECOMMENDED ACTIONS

        1. Dispatch a field inspector to verify and document defects within
           the recommended response window ({response_time}).
        2. Apply temporary asphalt patching for potholes deeper than 50 mm
           to restore immediate road safety.
        3. Seal cracks exceeding 3 mm in width with hot or cold bituminous
           sealant to prevent water ingress.
        4. Schedule full resurfacing assessment if the affected area exceeds
           15% of the lane width.

        TIMELINE & PRIORITY

        Based on a severity score of {score:.0f}/100, this case is classified as
        {severity_level} priority. The recommended response time is {response_time}.
        Work orders should be raised within 24 hours of this report.

        CLOSING NOTE

        If the {severity_level.lower()} severity classification cannot be addressed
        within the recommended timeframe, this incident should be escalated to the
        District Road Maintenance Supervisor. All repair activities must be recorded
        in the municipal asset management system upon completion.
    """).strip()

    return text, "template (no API key)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    severity_level: str,
    defect_count: int,
    breakdown: dict[str, int],
    score: float,
    area_pct: float,
    avg_confidence: float,
    response_time: str,
    location_hint: str = "Unknown location",
    notes: Optional[list[str]] = None,
) -> OfficialReport:
    """
    Generate an official road maintenance report.

    Args:
        severity_level:  e.g. "High"
        defect_count:    total number of defects detected
        breakdown:       {label: count} dict
        score:           numeric severity score (0–100)
        area_pct:        % of road surface covered by defects
        avg_confidence:  mean detection confidence (0–1)
        response_time:   recommended repair window string
        location_hint:   human description of location
        notes:           optional list of AI analysis notes

    Returns:
        OfficialReport instance.
    """
    if notes is None:
        notes = []

    provider = _get_provider()
    now = datetime.utcnow()
    report_id = f"SR-{now.strftime('%Y%m%d')}-{abs(hash(location_hint + str(score))) % 9000 + 1000}"
    generated_at = now.strftime("%Y-%m-%d %H:%M UTC")

    title = (
        f"Road Defect Report — {severity_level} Severity"
        f" | {now.strftime('%d %b %Y')}"
    )

    try:
        if provider == "groq":
            prompt = _build_prompt(
                severity_level, defect_count, breakdown, score,
                area_pct, avg_confidence, response_time, location_hint, notes
            )
            full_text, model_name = _call_groq(prompt)
        elif provider == "openai":
            prompt = _build_prompt(
                severity_level, defect_count, breakdown, score,
                area_pct, avg_confidence, response_time, location_hint, notes
            )
            full_text, model_name = _call_openai(prompt)
        else:
            full_text, model_name = _mock_report(
                severity_level, defect_count, breakdown, score, response_time, location_hint
            )

    except Exception as exc:
        print(f"[report_generator] LLM call failed ({exc}), falling back to mock.")
        full_text, model_name = _mock_report(
            severity_level, defect_count, breakdown, score, response_time, location_hint
        )
        provider = "mock"

    # Extract first ~2 sentences as executive summary
    sentences = [s.strip() for s in full_text.replace("\n", " ").split(".") if s.strip()]
    exec_summary = ". ".join(sentences[:2]) + "." if sentences else full_text[:200]

    return OfficialReport(
        title=title,
        executive_summary=exec_summary,
        full_text=full_text,
        generated_at=generated_at,
        report_id=report_id,
        provider=provider,
        model=model_name,
        location_hint=location_hint,
        severity_level=severity_level,
        defect_count=defect_count,
    )