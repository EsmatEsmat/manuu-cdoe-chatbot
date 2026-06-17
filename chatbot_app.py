# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:04:59 2026

@author: ismat
"""
import ssl
ssl.create_default_context = lambda *args, **kwargs: ssl._create_unverified_context(*args, **kwargs)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import os

from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from deep_translator import GoogleTranslator


# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="MAVIN - CDOE MANUU",
    page_icon="manuu_logo.png",  
    layout="centered"
)


# -----------------------------------
# SIDEBAR MODULAR PREVIEW TRIGGER
# -----------------------------------
st.sidebar.markdown("### 🖥️ Display Configuration")
ui_mode = st.sidebar.radio(
    "Select Application Interface Canvas Mode:",
    ["Full Screen Web App", "Floating Website Widget Preview"]
)


# -----------------------------------
# LOAD MODEL & FAQ DATA
# -----------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_faq():
    faq = pd.read_csv("master_faq.csv", encoding="latin1")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    faq["search_text"] = (
        faq["Main Question"].fillna("") + " " +
        faq["Alternate Questions"].fillna("") + " " +
        faq["Real Student Variants"].fillna("") + " " +
        faq["Intent"].fillna("") + " " +
        faq["Category"].fillna("") + " " +
        faq["Programme Group"].fillna("") + " " +
        faq["Programme Detail"].fillna("")
    )

    faq["search_text"] = faq["search_text"].apply(clean_text)
    return faq

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

model = load_model()
faq = load_faq()

faq_embeddings = model.encode(
    faq["search_text"].tolist(),
    convert_to_tensor=False
)


# -----------------------------------
# CORE BACKEND MATCHING ENGINE
# -----------------------------------

def handle_keyword_overrides(cleaned_question):
    tokens = cleaned_question.split()
    if len(tokens) <= 2:
        for keyword in ["mba", "bca", "bcom", "bed", "ma", "ba"]:
            if keyword in tokens:
                matches = faq[
                    faq["Main Question"].str.lower().str.contains(f"about {keyword}", na=False) |
                    faq["Intent"].str.lower().str.contains(f"{keyword} general", na=False)
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    return {
                        "answer": row["Answer"],
                        "matched_question": row["Main Question"],
                        "category": row.get("Category", "Programme Overview"),
                        "intent": row.get("Intent", "Direct Keyword Precision"),
                        "score": 1.000
                    }
    return None

def get_answer(user_question):
    cleaned_question = clean_text(user_question)
    direct_match = handle_keyword_overrides(cleaned_question)
    if direct_match:
        return direct_match

    question_embedding = model.encode([cleaned_question], convert_to_tensor=False)
    semantic_scores = cosine_similarity(question_embedding, faq_embeddings)[0]

    combined_scores = []
    for i, row_text in enumerate(faq["search_text"]):
        combined_scores.append(float(semantic_scores[i]))

    best_index = max(range(len(combined_scores)), key=combined_scores.__getitem__)
    best_score = combined_scores[best_index]
    row = faq.iloc[best_index]

    safety_check_score = (0.70 * best_score) + (0.30 * (fuzz.WRatio(cleaned_question, row["search_text"]) / 100))

    if safety_check_score < 0.45:
        return {
            "answer": "I am sorry, I can only answer MANUU CDOE related questions. Please refine your query with more specific terms.",
            "matched_question": "No confident match",
            "category": "-",
            "intent": "-",
            "score": round(float(safety_check_score), 3)
        }

    return {
        "answer": row["Answer"],
        "matched_question": row["Main Question"],
        "category": row["Category"] if "Category" in row else "-",
        "intent": row["Intent"] if "Intent" in row else "-",
        "score": round(float(best_score), 3)
    }

def save_log(user_query, result, original_urdu=""):
    log_file = "chat_logs.csv"
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Query": user_query if not original_urdu else f"{original_urdu} (Translated: {user_query})",
        "Bot Answer": result["answer"],
        "Matched Question": result["matched_question"],
        "Category": result["category"],
        "Intent": result["intent"],
        "Confidence Score": result["score"]
    }
    log_df = pd.DataFrame([log_data])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        log_df.to_csv(log_file, index=False, encoding="utf-8-sig")

def show_speech_button(answer_text):
    safe_answer = answer_text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    components.html(
        f"""
        <div style="margin:0; padding:0; background:transparent; overflow:hidden;">
            <button onclick="speakAnswer()" style="
                background-color:#00e676; color:white; border:none; padding:10px 16px;
                border-radius:8px; cursor:pointer; font-weight:600; font-size:15px; box-shadow: 0 2px 8px rgba(0,230,118,0.3);">
                🔊 Listen to Answer / جواب سنیں
            </button>
        </div>
        <script>
        function speakAnswer() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{safe_answer}');
            msg.lang = 'en-IN'; msg.rate = 0.9;
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """, height=50
    )


# -----------------------------------
# BRIGHT & VIBRANT VISUAL ENGINE
# -----------------------------------

widget_css = ""
if ui_mode == "Floating Website Widget Preview":
    widget_css = """
    .block-container {
        max-width: 430px !important;
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 24px !important;
        padding: 24px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.4) !important;
        margin: 30px auto !important;
        border: 2px solid rgba(255,255,255,0.25);
        backdrop-filter: blur(15px);
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.cdnfonts.com/css/jameel-noori-nastaleeq');
    
    /* Vibrant, Ultra-Bright Core Ambient Background Gradient */
    .stApp {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important;
    }}
    
    .block-container {{
        max-width: 750px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }}
    
    /* Absolute Font Hierarchy Fix */
    body, p, span, div, label {{
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }}
    
    /* Forcing Jameel Noori Nastaleeq explicitly into text blocks and text inputs */
    .urdu-text, [lang="ur"], .stAlert p, .custom-label span, div[data-baseweb="input"] input {{
        font-family: 'Jameel Noori Nastaleeq', 'Urdu Typesetting', 'Nastaliq', sans-serif !important;
    }}
    
    /* Direct targeting of input box to render typed Urdu perfectly */
    div[data-baseweb="input"] input {{
        font-size: 22px !important;
        line-height: 1.8 !important;
        direction: auto !important;
    }}
    
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-left: 5px solid #00e676 !important;
        border-radius: 14px;
        backdrop-filter: blur(10px);
    }}
    div.stTextInput > div > div > input {{
        border-radius: 14px; 
        border: 2px solid #1a73e8; 
        padding: 14px; 
        background-color: #ffffff !important;
        color: #1a252f !important;
    }}
    .answer-box {{
        background-color: #ffffff !important; padding: 26px; border-radius: 18px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); border-left: 6px solid #00e676; margin: 20px 0;
    }}
    .answer-title {{ color: #1e3c72 !important; font-weight: 700; font-size: 19px; }}
    .answer-text {{ font-size: 16px; color: #1f2937 !important; line-height: 1.65; }}
    
    /* RESTORED ANIMATION FOR THE MAVIN ROBOT ORB FIGURE */
    .avatar-container {{
        position: relative;
        width: 80px;
        height: 80px;
        margin: 20px auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .core-glow-orb {{
        width: 34px;
        height: 34px;
        background: radial-gradient(circle, #ffffff 0%, #00e676 80%);
        border-radius: 50%;
        box-shadow: 0 0 25px #00e676, 0 0 45px #1a73e8;
        animation: orbPulse 2.5s infinite ease-in-out;
    }}
    .satellite-orbit-ring {{
        position: absolute;
        width: 74px;
        height: 24px;
        border: 3px solid rgba(255, 255, 255, 0.9);
        border-radius: 50%;
        transform: rotateX(65deg) rotateY(15deg);
        animation: ringOrbitSpin 4s infinite linear;
        box-shadow: 0 0 8px #ffffff;
    }}
    
    /* Escaping percent characters properly inside dynamic strings */
    @keyframes orbPulse {{
        0%%, 100%% {{ transform: scale(1); box-shadow: 0 0 20px #00e676, 0 0 35px #1e3c72; }}
        50%% {{ transform: scale(1.15); box-shadow: 0 0 35px #ffffff, 0 0 55px #00e676; }}
    }}
    @keyframes ringOrbitSpin {{
        0%% {{ transform: rotateX(65deg) rotateY(15deg) rotateZ(0deg); }}
        100%% {{ transform: rotateX(65deg) rotateY(15deg) rotateZ(360deg); }}
    }}
    
    {widget_css}
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# GEOMETRICALLY CENTERED BRANDING HEADER
# -----------------------------------

st.markdown(
"""
<div style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 12px;">
    <img src="https://raw.githubusercontent.com/your-username/your-repo/main/manuu_logo.png" 
         onerror="this.src='https://upload.wikimedia.org/wikipedia/en/3/3b/Maulana_Azad_National_Urdu_University_Logo.png';" 
         style="width: 135px; height: auto;" />
</div>

<div style='text-align: center; margin-top: 5px; margin-bottom: 25px;'>
<h2 style='color: #ffffff !important; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 22px; margin-bottom: 6px; letter-spacing: 0.5px;'>
MAULANA AZAD NATIONAL URDU UNIVERSITY
</h2>
<p style='color: #ffffff !important; font-family: "Helvetica Neue", Arial, sans-serif; font-size: 18px; font-weight: 500; letter-spacing: 0.5px; margin-top: 0; margin-bottom: 10px;'>
Centre for Distance and Online Education
</p>

<!-- MAVIN VISUAL KINETIC ORB INTERFACE -->
<div class="avatar-container">
    <div class="core-glow-orb"></div>
    <div class="satellite-orbit-ring"></div>
</div>

<div style="margin-bottom: 10px;">
<div style="font-family: 'Jameel Noori Nastaleeq', 'Urdu Typesetting', sans-serif !important; font-weight: 500; font-size: 58px; color: #00e676; line-height: 1.1; direction: rtl;" lang="ur">
معاوِن
</div>
<div style='color: #ffffff !important; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 900; font-size: 46px; margin-top: -5px; margin-bottom: 2px; letter-spacing: 1px;'>
MAVIN
</div>
</div>
<p style='color: rgba(255,255,255,0.95) !important; font-size: 16px; margin-top: 2px; font-weight: 500; letter-spacing: 0.5px;'>
(MANUU Virtual Interface)
</p>
<div style='height: 3px; width: 160px; background: linear-gradient(90deg, #ffffff, #00e676); margin: 15px auto; border-radius: 2px;'></div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="stAlert" style="padding: 15px; margin-bottom: 20px;">
        <span style="color:#ffffff;">💡</span> <strong style="color:#ffffff;">Language Support:</strong> <span style="color:#ffffff;">You can type your questions comfortably in English or</span> <span class="urdu-text" style="font-family:'Jameel Noori Nastaleeq', sans-serif !important; font-size:24px; color:#00e676; vertical-align:middle;">اردو (اردو)</span>
    </div>
    """, 
    unsafe_allow_html=True
)

if "show_analytics" not in st.session_state:
    st.session_state.show_analytics = False


# -----------------------------------
# HARD-FORCED URDU INTERFACE LABELS
# -----------------------------------

st.markdown(
    """
    <label class="custom-label" style="color: #ffffff !important; font-size: 16px; font-weight: 500; margin-bottom: 8px; display: block;">
        How can MAVIN assist you today? / 
        <span style="font-family:'Jameel Noori Nastaleeq', 'Urdu Typesetting', sans-serif !important; font-size:26px; color:#00e676; vertical-align: middle; direction: rtl;">
            میں آپ کی کیا مدد کر سکتا ہوں؟
        </span>
    </label>
    """, 
    unsafe_allow_html=True
)
student_query = st.text_input("", placeholder="Type here in English or Urdu...", label_visibility="collapsed")


# -----------------------------------
# EXECUTION CONTEXT TRACKING
# -----------------------------------

if student_query:
    if student_query.strip() == "manuuadmin2026":
        st.session_state.show_analytics = not st.session_state.show_analytics
        st.success(f"Diagnostics View Visibility Toggled to: {st.session_state.show_analytics}")
    else:
        processed_query = student_query
        urdu_detected = is_urdu(student_query)
        
        with st.spinner("✨ MAVIN is processing..."):
            if urdu_detected:
                try:
                    processed_query = GoogleTranslator(source='ur', target='en').translate(student_query)
                except Exception as e:
                    pass
            
            result = get_answer(processed_query)
            save_log(processed_query, result, original_urdu=student_query if urdu_detected else "")

        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-title">🤖 MAVIN:</div>
                <div class="answer-text">{result["answer"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        show_speech_button(result["answer"])

        if st.session_state.show_analytics:
            with st.expander("📊 Technical Analytics (Office Evaluation Mode Only)", expanded=True):
                st.markdown(f"**Confidence Match Score:** `{result['score']}`")
                st.markdown(f"**Mapped Database Intent:** `{result['intent']}`")
                st.markdown(f"**Category:** `{result['category']}`")
                st.markdown(f"**Reference Master Question:** *\"{result['matched_question']}\"*")
                if urdu_detected:
                    st.markdown(f"**Original Urdu Script Query:** *\"{student_query}\"*")
