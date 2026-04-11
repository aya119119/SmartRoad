/**
 * frontend/assets/app.js
 * ========================
 * Main JavaScript file for SmartRoad frontend.
 * Handles file uploads, API calls, and result display.
 */

// ─────────────────────────────────────────────────────────────────────────────
// DOM Elements
// ─────────────────────────────────────────────────────────────────────────────

const uploadBtn = document.getElementById("uploadBtn");
const sampleBtn = document.getElementById("sampleBtn");
const demoBtn = document.getElementById("demoBtn");
const centerBtn = document.getElementById("centerBtn");

const uploadModal = document.getElementById("uploadModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const fileInput = document.getElementById("fileInput");
const locationInput = document.getElementById("locationInput");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingOverlay = document.getElementById("loadingOverlay");
const resultsSection = document.getElementById("resultsSection");
const backBtn = document.getElementById("backBtn");
const submitReportBtn = document.getElementById("submitReportBtn");

const annotatedImage = document.getElementById("annotatedImage");
const defectCount = document.getElementById("defectCount");
const confidence = document.getElementById("confidence");
const inferenceTime = document.getElementById("inferenceTime");
const severityLevel = document.getElementById("severityLevel");
const severityScore = document.getElementById("severityScore");
const severityBar = document.getElementById("severityBar");
const responseTime = document.getElementById("responseTime");
const reportContent = document.getElementById("reportContent");

// State
let currentAnalysis = null;
let selectedFile = null;

// ─────────────────────────────────────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────────────────────────────────────

uploadBtn.addEventListener("click", () => {
    uploadModal.classList.remove("hidden");
});

demoBtn.addEventListener("click", () => {
    uploadModal.classList.remove("hidden");
});

centerBtn.addEventListener("click", () => {
    uploadModal.classList.remove("hidden");
});

sampleBtn.addEventListener("click", async () => {
    // Load a sample image from data/samples/
    await loadSampleImage();
});

closeModalBtn.addEventListener("click", () => {
    uploadModal.classList.add("hidden");
    selectedFile = null;
    fileInput.value = "";
    analyzeBtn.disabled = true;
});

backBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    uploadModal.classList.remove("hidden");
    currentAnalysis = null;
    selectedFile = null;
    fileInput.value = "";
});

fileInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    analyzeBtn.disabled = !selectedFile;
    
    if (selectedFile) {
        analyzeBtn.disabled = false;
        // Visual feedback
        analyzeBtn.textContent = `Analyze ${selectedFile.name}`;
    }
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    await analyzeImage();
});

submitReportBtn.addEventListener("click", async () => {
    if (!currentAnalysis) return;
    await submitReport();
});

// Drag and drop
fileInput.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.parentElement.classList.add("bg-primary/10");
});

fileInput.addEventListener("dragleave", () => {
    fileInput.parentElement.classList.remove("bg-primary/10");
});

fileInput.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.parentElement.classList.remove("bg-primary/10");
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        selectedFile = files[0];
        analyzeBtn.disabled = false;
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// API Functions
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

/**
 * Send image to backend for analysis
 */
async function analyzeImage() {
    if (!selectedFile) return;

    try {
        loadingOverlay.classList.remove("hidden");
        uploadModal.classList.add("hidden");

        // Create FormData for file upload
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("location_hint", locationInput.value || "Unknown Location");

        // Call API
        const response = await fetch(`${API_BASE}/api/detect`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const analysis = await response.json();
        currentAnalysis = analysis;

        // Display results
        displayResults(analysis);
        resultsSection.classList.remove("hidden");

    } catch (error) {
        console.error("Analysis error:", error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        loadingOverlay.classList.add("hidden");
    }
}

/**
 * Load a sample image from the data/samples folder
 */
async function loadSampleImage() {
    try {
        // In production, you'd list available samples from the server
        // For now, create a placeholder or fetch a sample
        const response = await fetch(`${API_BASE}/api/test`);
        if (!response.ok) throw new Error("Sample not available");
        
        alert("Sample analysis feature coming soon!");
    } catch (error) {
        console.error("Sample load error:", error);
        alert("Could not load sample image");
    }
}

/**
 * Submit report to municipality
 */
async function submitReport() {
    if (!currentAnalysis) return;

    try {
        submitReportBtn.disabled = true;
        submitReportBtn.innerHTML = '<span class="loading-spinner" style="width:20px; height:20px; border-width:2px;"></span> Submitting...';

        const response = await fetch(`${API_BASE}/api/report`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                report_id: currentAnalysis.report?.report_id || `report_${Date.now()}`,
                municipality_email: "municipality@maroc.gov.ma",
                additional_notes: locationInput.value,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();

        // Show success message
        alert(`✓ Report Submitted!\n\nSubmission ID: ${result.submission_id}\n\nThe municipal authority will review this report shortly.`);

        submitReportBtn.innerHTML = '<span class="material-symbols-outlined" data-icon="send">send</span> Submit to Municipality';
        submitReportBtn.disabled = false;

    } catch (error) {
        console.error("Submit error:", error);
        alert(`Submission failed: ${error.message}`);
        submitReportBtn.innerHTML = '<span class="material-symbols-outlined" data-icon="send">send</span> Submit to Municipality';
        submitReportBtn.disabled = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Display Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Display analysis results
 */
function displayResults(analysis) {
    // Annotated image
    if (analysis.annotated_image) {
        annotatedImage.src = analysis.annotated_image;
    }

    // Detection stats
    const detection = analysis.detection;
    defectCount.textContent = detection.count;
    confidence.textContent = `${Math.round(detection.highest_confidence * 100)}%`;
    inferenceTime.textContent = `${detection.inference_time_ms}ms`;

    // Severity assessment
    const severity = analysis.severity;
    severityLevel.textContent = severity.level;
    severityScore.textContent = severity.score.toFixed(0);
    
    // Color the severity bar based on level
    const severityColors = {
        "Low": "bg-green-500",
        "Moderate": "bg-yellow-500",
        "High": "bg-orange-500",
        "Critical": "bg-red-500",
    };
    
    severityBar.className = `h-full rounded-full transition-all ${severityColors[severity.level] || "bg-primary"}`;
    severityBar.style.width = `${severity.score}%`;
    
    // Set text color for severity level
    const severityTextColors = {
        "Low": "text-green-500",
        "Moderate": "text-yellow-600",
        "High": "text-orange-500",
        "Critical": "text-red-500",
    };
    severityLevel.className = `font-bold ${severityTextColors[severity.level] || "text-secondary"}`;
    
    // Response time
    responseTime.textContent = `Recommended Response: ${severity.recommended_response_time}`;

    // Defect breakdown
    let breakdownText = "Defects detected: ";
    if (severity.breakdown && Object.keys(severity.breakdown).length > 0) {
        breakdownText += Object.entries(severity.breakdown)
            .map(([type, count]) => `${count} ${type}(s)`)
            .join(", ");
    } else {
        breakdownText += "None";
    }

    // Display notes
    const notesHtml = severity.notes
        .map(note => `<li class="ml-4">• ${note}</li>`)
        .join("");

    // Report content
    const report = analysis.report;
    reportContent.innerHTML = `
        <div class="bg-stone-800 p-4 rounded-lg mb-4">
            <p class="text-white/90"><strong>Report ID:</strong> ${report.report_id}</p>
            <p class="text-white/90"><strong>Generated:</strong> ${new Date(report.generated_at).toLocaleString()}</p>
            <p class="text-white/90"><strong>Provider:</strong> ${report.provider}</p>
            <p class="text-white/90"><strong>Model:</strong> ${report.model}</p>
        </div>
        
        <div class="mb-4">
            <h4 class="font-bold text-white mb-2">Executive Summary</h4>
            <p>${report.executive_summary}</p>
        </div>
        
        <div class="mb-4">
            <h4 class="font-bold text-white mb-2">Breakdown</h4>
            <p>${breakdownText}</p>
        </div>
        
        <div class="mb-4">
            <h4 class="font-bold text-white mb-2">Observations</h4>
            <ul class="text-white/80">
                ${notesHtml}
            </ul>
        </div>
        
        <div class="bg-stone-800 p-4 rounded-lg border border-white/10">
            <h4 class="font-bold text-white mb-2">Full Report</h4>
            <p class="whitespace-pre-wrap">${report.full_text}</p>
        </div>
    `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    console.log("[SmartRoad] Frontend initialized");
    
    // Check if backend is running
    fetch(`${API_BASE}/health`)
        .then(r => r.json())
        .then(data => {
            console.log("[SmartRoad] Backend health:", data);
            if (data.model_loaded) {
                console.log("✓ Model is loaded and ready");
            }
        })
        .catch(err => {
            console.warn("[SmartRoad] Backend not available:", err);
            alert("Warning: Backend server is not running. Make sure to start the FastAPI server.");
        });
});
