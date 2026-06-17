"""
AI-Powered Payment Soundbox
============================
A Streamlit app that uses OCR to detect payment success/failure from a
captured image of a digital wallet screen and plays an audio notification.

Requirements:
    pip install streamlit easyocr pillow numpy

Run:
    streamlit run payment_soundbox.py --server.address=0.0.0.0 --server.port=8501
"""

import streamlit as st
from PIL import Image
import numpy as np
import base64
import io

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payment Soundbox",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS: mobile-friendly, clean dark theme ─────────────────────────────
st.markdown(
    """
    <style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Hide hamburger / footer */
    #MainMenu, footer { visibility: hidden; }

    /* Center container on desktop, full-width on mobile */
    .block-container {
        max-width: 480px;
        margin: 0 auto;
        padding: 1rem 1rem 4rem;
    }

    /* Header */
    .sb-header {
        text-align: center;
        padding: 1.4rem 0 0.4rem;
    }
    .sb-header .icon { font-size: 3rem; }
    .sb-header h1 {
        margin: 0.2rem 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .sb-header p {
        color: #8b949e;
        font-size: 0.88rem;
        margin: 0.25rem 0 0;
    }

    /* Step pill */
    .step-pill {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        color: #8b949e;
        margin: 1rem 0 0.4rem;
        font-weight: 500;
    }

    /* Result banners */
    .result-success {
        background: #0d2b1a;
        border: 2px solid #2ea043;
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-success .big-icon { font-size: 3rem; }
    .result-success h2 { color: #3fb950; margin: 0.4rem 0 0.2rem; font-size: 1.5rem; }
    .result-success p  { color: #8b949e; margin: 0; font-size: 0.9rem; }

    .result-failure {
        background: #2d1011;
        border: 2px solid #f85149;
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-failure .big-icon { font-size: 3rem; }
    .result-failure h2 { color: #f85149; margin: 0.4rem 0 0.2rem; font-size: 1.5rem; }
    .result-failure p  { color: #8b949e; margin: 0; font-size: 0.9rem; }

    /* OCR text box */
    .ocr-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
        color: #8b949e;
        font-family: 'JetBrains Mono', monospace;
        white-space: pre-wrap;
        word-break: break-word;
        margin-top: 0.6rem;
        max-height: 160px;
        overflow-y: auto;
    }

    /* Divider */
    .sb-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.2rem 0;
    }

    /* Streamlit camera widget label */
    label[data-testid="stWidgetLabel"] {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
    }

    /* Spinner color */
    .stSpinner > div { border-top-color: #2ea043 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="sb-header">
        <div class="icon">📳</div>
        <h1>Payment Soundbox</h1>
        <p>AI-powered payment verification for shopkeepers</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

# ── Instructions ──────────────────────────────────────────────────────────────
st.markdown('<span class="step-pill">📋 How to use</span>', unsafe_allow_html=True)
st.markdown(
    """
    1. Ask the customer to show their **payment confirmation screen**
    2. Tap **"Take Photo"** below to capture it
    3. Wait a moment — the AI will read the screen and alert you
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)


# ── Utility: generate a short pure-tone WAV in memory (no external file) ─────
def _generate_success_wav() -> bytes:
    """
    Build a minimal WAV file containing a two-tone 'ding ding' success chime.
    Uses only numpy — no external audio files required.
    Frequencies: 880 Hz (A5) followed by 1046.5 Hz (C6), each 0.22 s.
    """
    sample_rate = 22050
    note_samples = int(sample_rate * 0.22)    # samples per note
    fade_samples = int(sample_rate * 0.04)     # fade-out tail
    t = np.linspace(0, 0.22, note_samples, endpoint=False)

    def sine_note(freq: float) -> np.ndarray:
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Apply a quick exponential decay envelope
        envelope = np.exp(-t * 8)
        # Short linear fade-out at the tail to avoid clicks
        wave *= envelope
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        return (wave * 32767).astype(np.int16)

    note1 = sine_note(880.0)    # A5 — bright, clear
    silence = np.zeros(int(sample_rate * 0.04), dtype=np.int16)
    note2 = sine_note(1046.5)   # C6 — higher resolution tone

    pcm = np.concatenate([note1, silence, note2])

    # Pack into a WAV byte string manually (no soundfile/scipy needed)
    num_samples  = len(pcm)
    num_channels = 1
    bits_per_sample = 16
    byte_rate    = sample_rate * num_channels * bits_per_sample // 8
    block_align  = num_channels * bits_per_sample // 8
    data_size    = num_samples * block_align

    header = (
        b'RIFF'
        + (36 + data_size).to_bytes(4, 'little')
        + b'WAVE'
        + b'fmt '
        + (16).to_bytes(4, 'little')          # sub-chunk size
        + (1).to_bytes(2, 'little')            # PCM format
        + num_channels.to_bytes(2, 'little')
        + sample_rate.to_bytes(4, 'little')
        + byte_rate.to_bytes(4, 'little')
        + block_align.to_bytes(2, 'little')
        + bits_per_sample.to_bytes(2, 'little')
        + b'data'
        + data_size.to_bytes(4, 'little')
    )
    return header + pcm.tobytes()


def play_success_sound():
    """Embed a base64-encoded WAV as an autoplay <audio> tag."""
    wav_bytes  = _generate_success_wav()
    b64_audio  = base64.b64encode(wav_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# ── Keyword lists for payment verification ────────────────────────────────────
SUCCESS_KEYWORDS = [
    # English success phrases
    "successful", "success", "approved", "confirmed",
    "received", "complete", "completed", "done",
    "paid", "payment done", "transaction complete",
    "debited", "credited", "transferred",
    "congratulations", "thank you",
    # Common payment app phrases
    "money sent", "amount sent", "payment sent",
    "transaction successful",
    # Currency markers (Pakistan + international)
    "pkr", "rs.", "rs ", "₨", "usd", "inr", "৳",
    # Urdu / Roman Urdu (transliterated) common in Pakistani apps
    "bheja gaya", "raseed", "kamyab",
]

FAILURE_INDICATORS = [
    "failed", "failure", "declined", "rejected",
    "error", "invalid", "timeout", "insufficient",
    "not found", "try again",
]


# ── OCR loader (cached so the model is only loaded once) ─────────────────────
@st.cache_resource(show_spinner="Loading OCR engine (first time only)…")
def load_ocr_reader():
    """
    Initialise EasyOCR with English and Urdu language packs.
    gpu=False keeps it universally compatible; set gpu=True if you have CUDA.
    """
    try:
        import easyocr
        # 'en' covers English; 'ur' covers Urdu for Pakistani payment apps
        reader = easyocr.Reader(["en", "ur"], gpu=False, verbose=False)
        return reader, None
    except ImportError:
        return None, "easyocr"


def extract_text_easyocr(image: Image.Image, reader) -> str:
    """Run EasyOCR on a PIL image and return all detected text as one string."""
    img_array = np.array(image.convert("RGB"))
    results   = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


# ── Verification logic ────────────────────────────────────────────────────────
def verify_payment(raw_text: str) -> tuple[bool, list[str]]:
    """
    Scan OCR text for success keywords.

    Returns:
        (is_success: bool, matched_keywords: list[str])

    Logic:
        1. Convert text to lowercase for case-insensitive matching.
        2. Check for any failure indicator first — explicit failure overrides.
        3. Then check for success keywords.
    """
    text_lower = raw_text.lower()

    # Explicit failure check (hard override)
    for kw in FAILURE_INDICATORS:
        if kw in text_lower:
            return False, [kw]

    # Success keyword scan
    matched = [kw for kw in SUCCESS_KEYWORDS if kw in text_lower]
    if matched:
        return True, matched

    return False, []


# ── Camera input section ──────────────────────────────────────────────────────
st.markdown('<span class="step-pill">📸 Step 1 — Capture screen</span>', unsafe_allow_html=True)

captured_image = st.camera_input(
    label="Point camera at the customer's payment screen",
    help="Make sure the full confirmation message is visible before taking the photo.",
)

# ── Process captured image ────────────────────────────────────────────────────
if captured_image is not None:
    # Load OCR engine (cached after first load)
    ocr_reader, missing_lib = load_ocr_reader()

    if missing_lib:
        st.error(
            f"⚠️ Required library **{missing_lib}** is not installed. "
            f"Run `pip install {missing_lib}` and restart the app."
        )
        st.stop()

    # Open the image
    pil_image = Image.open(captured_image)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">🔍 Step 2 — Reading text…</span>', unsafe_allow_html=True)

    with st.spinner("Scanning payment screen with AI OCR…"):
        try:
            raw_ocr_text = extract_text_easyocr(pil_image, ocr_reader)
        except Exception as exc:
            st.error(f"OCR error: {exc}")
            st.stop()

    # ── Show the raw OCR output (collapsible, helpful for debugging) ──────────
    with st.expander("📄 Raw OCR text (tap to expand)", expanded=False):
        if raw_ocr_text.strip():
            st.markdown(
                f'<div class="ocr-box">{raw_ocr_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ocr-box"><em>No text detected in image.</em></div>',
                unsafe_allow_html=True,
            )

    # ── Verify payment ────────────────────────────────────────────────────────
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<span class="step-pill">✅ Step 3 — Verdict</span>', unsafe_allow_html=True)

    is_paid, matched_kws = verify_payment(raw_ocr_text)

    if is_paid:
        # ── SUCCESS ───────────────────────────────────────────────────────────
        play_success_sound()   # autoplay ding-ding chime

        matched_display = ", ".join(f'"{k}"' for k in matched_kws[:4])
        st.markdown(
            f"""
            <div class="result-success">
                <div class="big-icon">✅</div>
                <h2>Payment Received!</h2>
                <p>Detected: {matched_display}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()

    else:
        # ── FAILURE / UNVERIFIED ──────────────────────────────────────────────
        # No audio is injected — silence is guaranteed.
        reason = (
            f'Keyword "{matched_kws[0]}" detected.'
            if matched_kws
            else "No payment confirmation keyword found."
        )
        st.markdown(
            f"""
            <div class="result-failure">
                <div class="big-icon">❌</div>
                <h2>Not Verified</h2>
                <p>{reason} Please ask the customer to show the full confirmation screen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Reset hint ────────────────────────────────────────────────────────────
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.info("📷 Tap **'Clear photo'** above the image to scan a new payment.", icon="ℹ️")

else:
    # ── Idle state: show sample keyword list ─────────────────────────────────
    with st.expander("🔑 Keywords this app detects", expanded=False):
        kw_cols = st.columns(2)
        half    = len(SUCCESS_KEYWORDS) // 2
        with kw_cols[0]:
            for kw in SUCCESS_KEYWORDS[:half]:
                st.markdown(f"- `{kw}`")
        with kw_cols[1]:
            for kw in SUCCESS_KEYWORDS[half:]:
                st.markdown(f"- `{kw}`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#484f58;font-size:0.76rem;">'
    "AI Payment Soundbox · Built with Streamlit + EasyOCR"
    "</p>",
    unsafe_allow_html=True,
)
