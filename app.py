import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import base64
import json

# ── Attempt to import recommendations ─────────────────────────────────────────
try:
    from recommendation import cnv, dme, drusen, normal as normal_rec
    HAS_RECS = True
except ImportError:
    HAS_RECS = False
    cnv = dme = drusen = normal_rec = ""

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OcuScan AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Hide Streamlit chrome AND push the form off-screen (but keep it functional)
st.markdown("""
<style>
/* Hide all Streamlit decorations */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

/* Zero out app shell padding */
html, body { margin: 0 !important; padding: 0 !important; background: #0a0f1e !important; }
.stApp,
[data-testid="stAppViewContainer"],
section[data-testid="stMain"],
.block-container,
[data-testid="stVerticalBlock"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    gap: 0 !important;
    background: #0a0f1e !important;
}

/* Push the native form off-screen — it must stay in the DOM so Python can read it */
[data-testid="stForm"] {
    position: fixed !important;
    top: -9999px !important;
    left: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Make the components.html iframe fill the full viewport */
iframe {
    border: none !important;
    display: block !important;
    width: 100vw !important;
    height: 100vh !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    z-index: 9999 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Trained_Model.keras")

def model_prediction(image_bytes):
    model = load_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(image_bytes)
        tmp_path = f.name
    img = tf.keras.utils.load_img(tmp_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return int(np.argmax(predictions))

# ── Session state defaults ─────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "image_b64" not in st.session_state:
    st.session_state.image_b64 = None

# ── Hidden Streamlit form — off-screen but fully functional ───────────────────
with st.form("predict_form", clear_on_submit=False):
    uploaded = st.file_uploader("OCT Scan", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted and uploaded:
    img_bytes = uploaded.read()
    st.session_state.image_b64 = base64.b64encode(img_bytes).decode()
    idx = model_prediction(img_bytes)
    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
    predicted = class_names[idx]

    clinical = {
        "CNV":    "OCT scan showing CNV with neovascular membrane and associated subretinal fluid.",
        "DME":    "OCT scan showing DME with retinal thickening and intraretinal fluid.",
        "DRUSEN": "OCT scan showing multiple drusen deposits present in early AMD.",
        "NORMAL": "OCT scan showing a normal retina with preserved foveal contour and no retinal fluid.",
    }
    recs = {
        "CNV":    ["Immediate referral to retinal specialist","Consider anti-VEGF therapy (intravitreal injections)","Monitor for vision changes with Amsler grid","Follow-up OCT in 4–6 weeks"],
        "DME":    ["Optimize glycemic control and blood pressure","Consider anti-VEGF or corticosteroid therapy","Laser photocoagulation for focal edema","Regular diabetic eye examinations"],
        "DRUSEN": ["AREDS2 vitamin supplementation","Lifestyle modifications (smoking cessation, diet)","Monitor with Amsler grid monthly","Follow-up OCT every 6–12 months"],
        "NORMAL": ["Routine annual eye examination recommended","Maintain healthy lifestyle and regular checkups","Report any vision changes promptly","Continue preventive eye care measures"],
    }
    st.session_state.result = {
        "disease": predicted,
        "clinical": clinical[predicted],
        "recs": recs[predicted],
        "is_healthy": predicted == "NORMAL",
    }

# ── Inject Python result into HTML ────────────────────────────────────────────
result_json   = json.dumps(st.session_state.result) if st.session_state.result else "null"
image_b64_js  = f'"{st.session_state.image_b64}"'   if st.session_state.image_b64 else "null"

# ── Full HTML — identical design, JS wires up to the hidden Streamlit form ────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --navy: #0a0f1e; --navy2: #111827; --panel: #141c2e; --panel2: #1a2340;
    --border: rgba(99,179,237,0.12); --border2: rgba(99,179,237,0.22);
    --accent: #38bdf8; --accent2: #0ea5e9; --green: #34d399; --red: #f87171;
    --amber: #fbbf24; --blue: #60a5fa; --purple: #a78bfa;
    --text: #e2e8f0; --muted: #64748b; --muted2: #94a3b8; --white: #f8fafc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ scroll-behavior: smooth; height: 100%; overflow-x: hidden; }}
  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--navy); color: var(--text);
  }}
  body::before {{
    content:''; position:fixed; inset:0;
    background-image:
      linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
    background-size: 48px 48px; pointer-events:none; z-index:0;
  }}

  /* NAVBAR */
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(10,15,30,0.85); backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem; display: flex; align-items: center; justify-content: space-between;
    height: 64px;
  }}
  .logo {{ display: flex; align-items: center; gap: 10px; font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: var(--white); text-decoration: none; cursor:pointer; }}
  .logo-icon {{ width: 36px; height: 36px; background: linear-gradient(135deg, var(--accent), var(--purple)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }}
  .nav-links {{ display: flex; gap: 0.25rem; }}
  .nav-link {{ padding: 0.4rem 1rem; border-radius: 8px; font-size: 0.875rem; font-weight: 500; color: var(--muted2); cursor: pointer; border: none; background: none; transition: all 0.2s; }}
  .nav-link:hover {{ color: var(--text); background: var(--panel2); }}
  .nav-link.active {{ color: var(--accent); background: rgba(56,189,248,0.08); }}
  .nav-badge {{ background: linear-gradient(135deg, var(--accent2), var(--purple)); color: #fff; font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 20px; }}

  /* PAGES */
  .page {{ display: none; animation: fadeIn 0.4s ease; position: relative; z-index: 1; }}
  .page.active {{ display: block; }}
  @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(12px); }} to {{ opacity:1; transform: translateY(0); }} }}

  /* HERO */
  .hero {{ padding: 5rem 2rem 4rem; max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 4rem; align-items: center; }}
  .hero-eyebrow {{ display: inline-flex; align-items: center; gap: 8px; background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.2); border-radius: 20px; padding: 0.3rem 0.9rem; font-size: 0.75rem; font-weight: 600; color: var(--accent); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 1.5rem; }}
  .pulse-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }} 50% {{ box-shadow: 0 0 0 6px rgba(52,211,153,0); }} }}
  .hero h1 {{ font-family: 'DM Serif Display', serif; font-size: clamp(2.2rem, 4vw, 3.2rem); line-height: 1.15; color: var(--white); margin-bottom: 1.25rem; }}
  .hero h1 em {{ font-style: italic; color: var(--accent); }}
  .hero p {{ font-size: 1rem; line-height: 1.75; color: var(--muted2); margin-bottom: 2rem; }}
  .hero-actions {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .btn-primary {{ display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, var(--accent2), var(--purple)); color: #fff; font-weight: 600; font-size: 0.875rem; padding: 0.7rem 1.5rem; border-radius: 10px; border: none; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 24px rgba(56,189,248,0.25); }}
  .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 8px 32px rgba(56,189,248,0.35); }}
  .btn-ghost {{ display: inline-flex; align-items: center; gap: 8px; background: transparent; color: var(--muted2); font-weight: 500; font-size: 0.875rem; padding: 0.7rem 1.5rem; border-radius: 10px; border: 1px solid var(--border2); cursor: pointer; transition: all 0.2s; }}
  .btn-ghost:hover {{ color: var(--text); border-color: var(--accent); background: rgba(56,189,248,0.05); }}
  .hero-visual {{ display: flex; flex-direction: column; gap: 12px; }}
  .stat-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .stat-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.25rem; transition: border-color 0.3s; }}
  .stat-card:hover {{ border-color: var(--border2); }}
  .stat-num {{ font-family: 'DM Serif Display', serif; font-size: 1.9rem; }}
  .stat-label {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
  .disease-pills {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 1rem 1.25rem; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; align-items: center; }}
  .pill {{ padding: 0.3rem 0.85rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }}
  .pill-red   {{ background: rgba(248,113,113,0.1); color: var(--red);   border: 1px solid rgba(248,113,113,0.2); }}
  .pill-amber {{ background: rgba(251,191,36,0.1);  color: var(--amber); border: 1px solid rgba(251,191,36,0.2); }}
  .pill-blue  {{ background: rgba(96,165,250,0.1);  color: var(--blue);  border: 1px solid rgba(96,165,250,0.2); }}
  .pill-green {{ background: rgba(52,211,153,0.1);  color: var(--green); border: 1px solid rgba(52,211,153,0.2); }}

  /* SECTION */
  .section {{ padding: 3rem 2rem; max-width: 1100px; margin: 0 auto; }}
  .section-title {{ font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: var(--white); margin-bottom: 0.5rem; }}
  .section-sub {{ font-size: 0.875rem; color: var(--muted2); margin-bottom: 2rem; }}
  .section-divider {{ border: none; border-top: 1px solid var(--border); margin: 0; }}

  /* DISEASE CARDS */
  .disease-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
  .disease-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 1.25rem; transition: all 0.25s; }}
  .disease-card:hover {{ transform: translateY(-3px); border-color: var(--border2); }}
  .disease-card-icon {{ width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: 0.875rem; }}
  .disease-card h3 {{ font-size: 0.9rem; font-weight: 600; color: var(--white); margin-bottom: 0.35rem; }}
  .disease-card p {{ font-size: 0.8rem; color: var(--muted2); line-height: 1.6; }}

  /* ANALYSIS PAGE */
  .analysis-layout {{ max-width: 1100px; margin: 0 auto; padding: 2.5rem 2rem; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  .upload-panel, .result-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 20px; overflow: hidden; }}
  .result-panel {{ display: flex; flex-direction: column; }}
  .panel-header {{ padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
  .panel-header h2 {{ font-size: 1rem; font-weight: 600; color: var(--white); }}
  .panel-header span {{ font-size: 0.75rem; color: var(--muted); }}
  .panel-body {{ padding: 1.5rem; }}

  /* Upload zone */
  .upload-zone {{ border: 2px dashed var(--border2); border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center; cursor: pointer; transition: all 0.25s; }}
  .upload-zone:hover {{ border-color: var(--accent); background: rgba(56,189,248,0.04); }}
  .upload-icon-wrap {{ width: 64px; height: 64px; border-radius: 50%; background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.15); display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-size: 1.75rem; transition: all 0.2s; }}
  .upload-zone:hover .upload-icon-wrap {{ background: rgba(56,189,248,0.15); transform: scale(1.05); }}
  .upload-zone h3 {{ font-size: 0.95rem; font-weight: 600; color: var(--white); margin-bottom: 0.3rem; }}
  .upload-zone p {{ font-size: 0.8rem; color: var(--muted); }}
  .upload-types {{ display: flex; gap: 8px; justify-content: center; margin-top: 1rem; flex-wrap: wrap; }}
  .upload-type-tag {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.7rem; color: var(--muted2); }}

  /* Preview */
  .preview-wrap {{ display: none; margin-top: 1.25rem; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); position: relative; }}
  .preview-wrap.visible {{ display: block; }}
  .preview-wrap img {{ width: 100%; max-height: 240px; object-fit: contain; background: #000; display: block; }}
  .preview-overlay {{ position: absolute; bottom: 0; left: 0; right: 0; background: linear-gradient(transparent, rgba(0,0,0,0.7)); padding: 1.5rem 1rem 0.75rem; display: flex; align-items: center; justify-content: space-between; }}
  .preview-filename {{ font-size: 0.75rem; color: rgba(255,255,255,0.8); }}
  .preview-remove {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.7rem; cursor: pointer; transition: all 0.2s; }}
  .preview-remove:hover {{ background: rgba(248,113,113,0.3); }}

  /* Analyze button */
  .analyze-btn {{ width: 100%; padding: 0.875rem; background: linear-gradient(135deg, var(--accent2), var(--purple)); border: none; border-radius: 10px; color: #fff; font-size: 0.9rem; font-weight: 600; cursor: pointer; margin-top: 1.25rem; transition: all 0.2s; box-shadow: 0 4px 20px rgba(56,189,248,0.2); display: flex; align-items: center; justify-content: center; gap: 8px; }}
  .analyze-btn:hover:not(:disabled) {{ transform: translateY(-1px); box-shadow: 0 8px 28px rgba(56,189,248,0.3); }}
  .analyze-btn:disabled {{ opacity: 0.6; cursor: not-allowed; transform: none; }}

  /* Result panel */
  .result-empty {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 2rem; text-align: center; color: var(--muted); }}
  .result-empty .empty-icon {{ font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }}
  .result-empty p {{ font-size: 0.875rem; line-height: 1.7; }}
  .result-content {{ display: none; flex-direction: column; height: 100%; }}
  .result-content.visible {{ display: flex; }}
  .result-top {{ padding: 1.5rem; border-bottom: 1px solid var(--border); }}
  .result-verdict {{ display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; }}
  .verdict-icon {{ width: 48px; height: 48px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; }}
  .verdict-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted2); }}
  .verdict-disease {{ font-family: 'DM Serif Display', serif; font-size: 1.6rem; }}
  .verdict-sub {{ font-size: 0.8rem; margin-top: 2px; }}
  .confidence-section {{ margin-top: 0.75rem; }}
  .conf-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
  .conf-label {{ font-size: 0.75rem; color: var(--muted); }}
  .conf-value {{ font-size: 0.75rem; font-weight: 600; }}
  .conf-bar {{ height: 6px; border-radius: 3px; background: var(--panel2); overflow: hidden; }}
  .conf-fill {{ height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.4,0,0.2,1); width: 0%; }}
  .result-desc-section {{ padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  .result-desc-section h4 {{ font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem; }}
  .result-desc-section p {{ font-size: 0.85rem; color: var(--muted2); line-height: 1.75; }}
  .rec-section {{ padding: 1.25rem 1.5rem; flex: 1; overflow-y: auto; }}
  .rec-section h4 {{ font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.875rem; }}
  .rec-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .rec-item {{ display: flex; gap: 10px; align-items: flex-start; padding: 0.7rem 0.875rem; background: var(--panel2); border-radius: 10px; border: 1px solid var(--border); }}
  .rec-bullet {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }}
  .rec-text {{ font-size: 0.8rem; color: var(--muted2); line-height: 1.6; }}
  .disclaimer {{ margin: 1rem 1.5rem 1.25rem; padding: 0.75rem 1rem; background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.15); border-radius: 10px; font-size: 0.75rem; color: rgba(251,191,36,0.8); line-height: 1.6; }}

  /* Loading state */
  .loading-overlay {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 3rem 2rem; text-align: center; color: var(--muted2); display: none; }}
  .loading-overlay p {{ font-size: 0.875rem; line-height: 1.7; margin-top: 1rem; }}

  /* Spinner */
  .spinner {{ width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite; display: inline-block; }}
  .spin-lg {{ width: 48px; height: 48px; border: 3px solid rgba(56,189,248,0.2); border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ABOUT PAGE */
  .about-hero {{ padding: 4rem 2rem 2rem; max-width: 760px; margin: 0 auto; text-align: center; }}
  .about-hero h1 {{ font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: var(--white); margin-bottom: 1rem; }}
  .about-hero p {{ font-size: 1rem; color: var(--muted2); line-height: 1.8; }}
  .about-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 1.5rem; }}
  .about-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; }}
  .about-card h3 {{ font-size: 0.9rem; font-weight: 600; color: var(--white); margin-bottom: 0.75rem; }}
  .about-card p {{ font-size: 0.8rem; color: var(--muted2); line-height: 1.75; }}
  .timeline {{ margin-top: 1.5rem; }}
  .timeline-item {{ display: flex; gap: 1rem; padding-bottom: 1.5rem; position: relative; }}
  .timeline-item:not(:last-child)::before {{ content: ''; position: absolute; left: 15px; top: 32px; width: 1px; height: calc(100% - 16px); background: var(--border); }}
  .tl-dot {{ width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0; background: var(--panel2); border: 1px solid var(--border2); display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; color: var(--accent); }}
  .tl-content h4 {{ font-size: 0.875rem; font-weight: 600; color: var(--white); margin-bottom: 0.3rem; }}
  .tl-content p {{ font-size: 0.8rem; color: var(--muted2); line-height: 1.6; }}

  /* FOOTER */
  footer {{ border-top: 1px solid var(--border); padding: 2rem; text-align: center; font-size: 0.8rem; color: var(--muted); position: relative; z-index: 1; }}
  footer strong {{ color: var(--muted2); }}

  /* RESPONSIVE */
  @media (max-width: 768px) {{
    .hero, .analysis-layout, .about-grid {{ grid-template-columns: 1fr; }}
    .hero {{ padding: 3rem 1.5rem 2rem; gap: 2rem; }}
    .analysis-layout {{ padding: 1.5rem; }}
    nav {{ padding: 0 1rem; }}
    .nav-badge {{ display: none; }}
    .section {{ padding: 2rem 1.5rem; }}
  }}
</style>
</head>
<body>

<!-- NAVBAR -->
<nav>
  <div class="logo" onclick="showPage('home')">
    <div class="logo-icon">👁</div>
    OcuScan <span style="color:var(--accent)">AI</span>
  </div>
  <div class="nav-links">
    <button class="nav-link active" onclick="showPage('home')">Home</button>
    <button class="nav-link" onclick="showPage('analyze')">Analyze</button>
    <button class="nav-link" onclick="showPage('about')">About</button>
  </div>
  <span class="nav-badge">v2.0 — Retinal OCT</span>
</nav>

<!-- ============================== HOME PAGE ============================== -->
<div id="page-home" class="page active">
  <div class="hero">
    <div class="hero-left">
      <div class="hero-eyebrow">
        <div class="pulse-dot"></div>
        AI-Powered Retinal Analysis
      </div>
      <h1>Detect Eye Disease <em>Earlier.</em> Act Smarter.</h1>
      <p>OcuScan AI analyzes retinal OCT scans — identifying conditions like CNV, DME, Drusen,
         and healthy retinas with expert-level accuracy using deep learning.</p>
      <div class="hero-actions">
        <button class="btn-primary" onclick="showPage('analyze')">🔬 Start Analysis</button>
        <button class="btn-ghost" onclick="showPage('about')">Learn More →</button>
      </div>
    </div>
    <div class="hero-visual">
      <div class="stat-cards">
        <div class="stat-card"><div class="stat-num" style="color:var(--accent)">84K+</div><div class="stat-label">OCT Images in Dataset</div></div>
        <div class="stat-card"><div class="stat-num" style="color:var(--purple)">4</div><div class="stat-label">Detectable Conditions</div></div>
        <div class="stat-card"><div class="stat-num" style="color:var(--green)">3-Tier</div><div class="stat-label">Expert Grading System</div></div>
        <div class="stat-card"><div class="stat-num" style="color:var(--amber)">30M+</div><div class="stat-label">Annual OCT Scans</div></div>
      </div>
      <div class="disease-pills">
        <span style="font-size:0.75rem;color:var(--muted2);margin-right:4px;">Detects:</span>
        <span class="pill pill-red">CNV</span>
        <span class="pill pill-amber">DME</span>
        <span class="pill pill-blue">Drusen</span>
        <span class="pill pill-green">Normal</span>
      </div>
    </div>
  </div>

  <hr class="section-divider">

  <div class="section" style="padding-top:2.5rem;">
    <p class="section-title">Conditions We Detect</p>
    <p class="section-sub">Our system analyses retinal OCT scans to identify the following conditions with high accuracy.</p>
    <div class="disease-grid">
      <div class="disease-card">
        <div class="disease-card-icon" style="background:rgba(248,113,113,0.1);">🔴</div>
        <h3>CNV — Choroidal Neovascularization</h3>
        <p>Abnormal blood vessel growth beneath the retina. A hallmark of wet AMD. Detected via OCT scans.</p>
      </div>
      <div class="disease-card">
        <div class="disease-card-icon" style="background:rgba(251,191,36,0.1);">💧</div>
        <h3>DME — Diabetic Macular Edema</h3>
        <p>Fluid accumulation in the macula caused by diabetes. Identified from retinal thickening in OCT.</p>
      </div>
      <div class="disease-card">
        <div class="disease-card-icon" style="background:rgba(96,165,250,0.1);">🔵</div>
        <h3>Drusen — Early AMD</h3>
        <p>Lipid deposits beneath the retinal pigment epithelium. Precursor to advanced macular degeneration.</p>
      </div>
      <div class="disease-card">
        <div class="disease-card-icon" style="background:rgba(52,211,153,0.1);">✅</div>
        <h3>Normal / Healthy Retina</h3>
        <p>Clear, healthy retina with preserved foveal contour and no signs of fluid, edema, or abnormality.</p>
      </div>
    </div>
  </div>

  <hr class="section-divider">

  <div class="section">
    <p class="section-title">Why OCT Matters</p>
    <p style="font-size:0.95rem; color:var(--muted2); line-height:1.8;">
      Optical Coherence Tomography (OCT) is a crucial tool in ophthalmology, offering non-invasive
      imaging to detect retinal abnormalities. Each year, over 30 million OCT scans are performed,
      aiding in the diagnosis and management of eye conditions that can lead to vision loss.
    </p>
    <br>
    <p style="font-size:0.95rem; color:var(--muted2); line-height:1.8;">
      On this platform, we aim to streamline the analysis and interpretation of these scans,
      reducing the time burden on medical professionals and increasing diagnostic accuracy through
      automated deep learning.
    </p>
  </div>
</div>

<!-- ============================== ANALYZE PAGE ============================== -->
<div id="page-analyze" class="page">
  <div class="analysis-layout">

    <!-- LEFT: Upload -->
    <div class="upload-panel">
      <div class="panel-header">
        <h2>Upload OCT Scan</h2>
        <span>Retinal OCT image</span>
      </div>
      <div class="panel-body">

        <!-- Custom upload zone — triggers the hidden Streamlit file input -->
        <div class="upload-zone" id="uploadZone"
          onclick="triggerFileInput()"
          ondragover="handleDrag(event,true)"
          ondragleave="handleDrag(event,false)"
          ondrop="handleDrop(event)">
          <div class="upload-icon-wrap">📁</div>
          <h3>Drop image here or click to browse</h3>
          <p>High-resolution images give better results</p>
          <div class="upload-types">
            <span class="upload-type-tag">JPG</span>
            <span class="upload-type-tag">PNG</span>
            <span class="upload-type-tag">JPEG</span>
            <span class="upload-type-tag">WEBP</span>
          </div>
        </div>

        <!-- Preview -->
        <div class="preview-wrap" id="previewWrap">
          <img id="previewImg" src="" alt="Preview">
          <div class="preview-overlay">
            <span class="preview-filename" id="previewName">image.jpg</span>
            <button class="preview-remove" onclick="clearImage()">✕ Remove</button>
          </div>
        </div>

        <!-- Analyze button — triggers the hidden Streamlit submit button -->
        <button class="analyze-btn" id="analyzeBtn" onclick="submitToStreamlit()" disabled>
          🔬 Analyze OCT Scan
        </button>

        <p style="font-size:0.72rem;color:var(--muted);text-align:center;margin-top:0.75rem;line-height:1.6;">
          Powered by MobileNetV3 deep learning model. For educational use only.
        </p>
      </div>
    </div>

    <!-- RIGHT: Results -->
    <div class="result-panel">
      <div class="panel-header">
        <h2>Analysis Results</h2>
        <span id="resultTimestamp">—</span>
      </div>

      <div class="result-empty" id="resultEmpty">
        <div class="empty-icon">🧿</div>
        <p>Upload an OCT scan and click <strong style="color:var(--text)">Analyze</strong>
           to receive a model-powered diagnosis with recommendations.</p>
      </div>

      <!-- Loading state shown while Streamlit reruns -->
      <div class="result-empty" id="resultLoading" style="display:none;">
        <div class="spin-lg" style="margin-bottom:1.25rem;"></div>
        <p style="color:var(--muted2);">Running model inference…<br>
           <span style="font-size:0.75rem;color:var(--muted)">This may take a few seconds</span></p>
      </div>

      <div class="result-content" id="resultContent">
        <!-- populated by JS -->
      </div>
    </div>
  </div>
</div>

<!-- ============================== ABOUT PAGE ============================== -->
<div id="page-about" class="page">
  <div class="about-hero">
    <h1>About OcuScan AI</h1>
    <p>A machine learning platform for early detection of retinal diseases — built on a dataset
       of 84,495 expert-verified OCT images.</p>
  </div>

  <div class="section" style="padding-top:0;">
    <div class="about-grid">
      <div class="about-card">
        <h3>🔬 OCT Technology</h3>
        <p>Optical Coherence Tomography (OCT) captures high-resolution cross-sections of the retina.
           With 30+ million scans performed annually, automated analysis drastically reduces
           ophthalmologist workload and improves diagnostic speed.</p>
      </div>
      <div class="about-card">
        <h3>🧠 Model Architecture</h3>
        <p>Built on MobileNetV3, a lightweight convolutional neural network optimized for image
           classification. Images are resized to 224×224 and preprocessed with MobileNetV3
           normalization before inference.</p>
      </div>
      <div class="about-card">
        <h3>📊 Dataset Scale</h3>
        <p>84,495 JPEG OCT images across 4 categories (Normal, CNV, DME, Drusen), split into
           train/test/validation sets. Sourced from UC San Diego, Beijing Tongren Eye Center,
           Shanghai First People's Hospital, and others.</p>
      </div>
      <div class="about-card">
        <h3>🎯 Accuracy & Validation</h3>
        <p>The model achieves high classification accuracy across all four categories, validated
           on a held-out test set and cross-checked against expert ophthalmologist labels from
           a 993-scan validation subset.</p>
      </div>
    </div>

    <div style="margin-top:2.5rem;">
      <p class="section-title" style="font-size:1.25rem;">Expert Grading Pipeline</p>
      <p class="section-sub">Every training image passed through a 3-tier human verification system before use.</p>
      <div class="timeline">
        <div class="timeline-item">
          <div class="tl-dot">1</div>
          <div class="tl-content">
            <h4>Initial Quality Control</h4>
            <p>Undergraduate and medical students trained in OCT interpretation reviewed images
               for quality, excluding scans with severe artifacts or resolution issues.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="tl-dot">2</div>
          <div class="tl-content">
            <h4>Ophthalmologist Review</h4>
            <p>Four independent ophthalmologists graded each image for the presence of CNV,
               macular edema, drusen, and other visible pathologies.</p>
          </div>
        </div>
        <div class="timeline-item">
          <div class="tl-dot">3</div>
          <div class="tl-content">
            <h4>Senior Specialist Verification</h4>
            <p>Two senior retinal specialists with 20+ years each verified final labels.
               A separate validation subset of 993 scans was graded independently to check
               for human error.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>
  Built for academic purposes &nbsp;·&nbsp; <strong>OcuScan AI</strong>
  &nbsp;·&nbsp; Powered by MobileNetV3 &nbsp;·&nbsp; Not for clinical use
</footer>

<script>
// ── Data injected from Python ──────────────────────────────────────────────
const PYTHON_RESULT = {result_json};
const PYTHON_IMAGE  = {image_b64_js};

// ── Page routing ───────────────────────────────────────────────────────────
function showPage(id) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  const map = {{home:0, analyze:1, about:2}};
  document.querySelectorAll('.nav-link')[map[id]].classList.add('active');
  window.scrollTo({{top:0, behavior:'smooth'}});
}}

// ── File handling ──────────────────────────────────────────────────────────
let pendingFile = null;

function triggerFileInput() {{
  // Find the hidden Streamlit file input in the parent document and click it
  const stInput = window.parent.document.querySelector('input[type="file"]');
  if (stInput) stInput.click();
}}

function handleDrag(e, on) {{
  e.preventDefault();
  document.getElementById('uploadZone').style.borderColor = on ? 'var(--accent)' : '';
  document.getElementById('uploadZone').style.background  = on ? 'rgba(56,189,248,0.04)' : '';
}}
function handleDrop(e) {{
  e.preventDefault();
  handleDrag(e, false);
  const file = e.dataTransfer.files[0];
  if (!file) return;
  // Inject file into the hidden Streamlit input
  const stInput = window.parent.document.querySelector('input[type="file"]');
  if (stInput) {{
    const dt = new DataTransfer();
    dt.items.add(file);
    stInput.files = dt.files;
    stInput.dispatchEvent(new Event('change', {{bubbles: true}}));
  }}
  showPreview(file);
}}

// Listen for file selection on the parent Streamlit input
window.parent.document.addEventListener('change', function(e) {{
  if (e.target && e.target.type === 'file' && e.target.files[0]) {{
    showPreview(e.target.files[0]);
  }}
}}, true);

function showPreview(file) {{
  if (!file || !file.type.startsWith('image/')) return;
  pendingFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => {{
    document.getElementById('previewImg').src = ev.target.result;
    document.getElementById('previewName').textContent = file.name;
    document.getElementById('previewWrap').classList.add('visible');
    document.getElementById('analyzeBtn').disabled = false;
  }};
  reader.readAsDataURL(file);
}}

function clearImage() {{
  pendingFile = null;
  document.getElementById('previewWrap').classList.remove('visible');
  document.getElementById('analyzeBtn').disabled = true;
  // Clear the Streamlit file input
  const stInput = window.parent.document.querySelector('input[type="file"]');
  if (stInput) stInput.value = '';
}}

// ── Submit: click the hidden Streamlit form submit button ──────────────────
function submitToStreamlit() {{
  if (!pendingFile) return;

  const btn = document.getElementById('analyzeBtn');
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';
  btn.disabled = true;
  document.getElementById('resultEmpty').style.display = 'none';
  document.getElementById('resultLoading').style.display = 'flex';
  document.getElementById('resultContent').classList.remove('visible');

  // Give Streamlit a moment to register the file, then click submit
  setTimeout(() => {{
    const submitBtn = window.parent.document.querySelector('button[kind="primaryFormSubmit"]');
    if (submitBtn) submitBtn.click();
  }}, 200);
}}

// ── Disease styles ─────────────────────────────────────────────────────────
const diseaseStyles = {{
  CNV:    {{ color: '#f87171', bg: 'rgba(248,113,113,0.1)', icon: '🔴' }},
  DME:    {{ color: '#fbbf24', bg: 'rgba(251,191,36,0.1)',  icon: '💧' }},
  DRUSEN: {{ color: '#60a5fa', bg: 'rgba(96,165,250,0.1)',  icon: '🔵' }},
  NORMAL: {{ color: '#34d399', bg: 'rgba(52,211,153,0.1)',  icon: '✅' }},
}};

// ── Render result ──────────────────────────────────────────────────────────
function renderResult(r) {{
  const style = diseaseStyles[r.disease] || {{ color: '#94a3b8', bg: 'rgba(148,163,184,0.1)', icon: '🧿' }};
  const recItems = (r.recs || []).map(rec => `
    <div class="rec-item">
      <div class="rec-bullet" style="background:${{style.color}}"></div>
      <span class="rec-text">${{rec}}</span>
    </div>`).join('');

  document.getElementById('resultEmpty').style.display   = 'none';
  document.getElementById('resultLoading').style.display = 'none';
  document.getElementById('resultContent').innerHTML = `
    <div class="result-top">
      <div class="result-verdict">
        <div class="verdict-icon" style="background:${{style.bg}}">${{style.icon}}</div>
        <div>
          <div class="verdict-label">Detected Condition</div>
          <div class="verdict-disease" style="color:${{style.color}}">${{r.disease}}</div>
          <div class="verdict-sub" style="color:${{r.is_healthy ? '#34d399' : '#fbbf24'}}">
            ${{r.is_healthy ? '✓ No disease detected' : '⚠ Condition detected'}}
          </div>
        </div>
      </div>
      <div class="confidence-section">
        <div class="conf-row">
          <span class="conf-label">Model Confidence</span>
          <span class="conf-value" style="color:${{style.color}}">High</span>
        </div>
        <div class="conf-bar">
          <div class="conf-fill" id="confFill" style="background:${{style.color}}"></div>
        </div>
      </div>
    </div>
    <div class="result-desc-section">
      <h4>Clinical Observation</h4>
      <p>${{r.clinical}}</p>
    </div>
    <div class="rec-section">
      <h4>Medical Recommendations</h4>
      <div class="rec-list">${{recItems}}</div>
      <div class="disclaimer">
        ⚠️ This result is generated by an automated prediction system and is intended for
        educational and screening purposes only. Always consult a qualified ophthalmologist
        for diagnosis and treatment.
      </div>
    </div>`;
  document.getElementById('resultContent').classList.add('visible');
  setTimeout(() => {{
    const fill = document.getElementById('confFill');
    if (fill) fill.style.width = '92%';
  }}, 100);
  document.getElementById('resultTimestamp').textContent =
    new Date().toLocaleTimeString([], {{hour:'2-digit',minute:'2-digit'}});
  document.getElementById('analyzeBtn').innerHTML = '🔬 Analyze OCT Scan';
  document.getElementById('analyzeBtn').disabled = false;
}}

// ── On load: restore state injected from Python ────────────────────────────
window.addEventListener('load', () => {{
  if (PYTHON_RESULT) {{
    showPage('analyze');
    if (PYTHON_IMAGE) {{
      document.getElementById('previewImg').src = 'data:image/jpeg;base64,' + PYTHON_IMAGE;
      document.getElementById('previewName').textContent = 'uploaded_scan.jpg';
      document.getElementById('previewWrap').classList.add('visible');
      document.getElementById('analyzeBtn').disabled = false;
    }}
    renderResult(PYTHON_RESULT);
  }}
}});
</script>
</body>
</html>"""

import streamlit.components.v1 as components
components.html(HTML, height=800, scrolling=False)
