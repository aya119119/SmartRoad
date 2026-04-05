# SmartRoad — AI-Powered Street Defect Detection & Reporting System


##  Overview

**SmartRoad** is an end-to-end intelligent road monitoring system that automatically detects and classifies street defects from dashcam footage or static images, then uses a Large Language Model (LLM) to generate official reports and alert the relevant municipal authorities.

The system bridges the gap between raw visual data and actionable government response — turning a single road image into a complete, ready-to-send maintenance report in seconds.

---

##  Problem Statement

Road infrastructure defects such as potholes, cracks, and damaged street equipment (e.g. fallen lights, broken poles) pose serious safety risks to drivers and pedestrians. Current reporting systems rely on manual inspection or citizen complaints, which are slow, inconsistent, and often miss critical issues.

This project proposes an automated pipeline that:
- Continuously analyzes road footage from car-mounted cameras
- Detects and classifies defects in real time
- Scores their severity automatically
- Generates a formal report using an AI language model
- Notifies the appropriate government authority instantly

## Getting started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. Upload a video or image, enter a location, and generate a municipal report.

### Optional

- Set `OPENAI_API_KEY` or `GROQ_API_KEY` to use a real LLM for report generation.
- Provide latitude and longitude to place the alert on the map.

