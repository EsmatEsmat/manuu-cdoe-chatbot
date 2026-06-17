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
import json

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


# -----------------------------------
# HARD-FORCED IFRAME AUDIO ENGINE
# -----------------------------------
def show_speech_button(answer_text):
    # FIXED: Using json.dumps safely encodes all newlines, quotes, and backslashes for JS injection.
    safe_js_string = json.dumps(answer_text)
    
    html_template = '''
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    .speech-btn {
        background-color: #00e676; 
        border: none; 
        padding: 10px 18px;
        border-radius: 8px; 
        cursor: pointer; 
        font-size: 15px; 
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,230,118,0.3);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #ffffff !important;
    }
    .speech-btn span {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', 'Urdu Typesetting', sans-serif !important;
        font-size: 20px;
        font-weight: 500;
        color: #ffffff !important;
    }
    </style>
    </head>
    <body style="margin:0; padding:0; background:transparent;">
        <button class="speech-btn" onclick="speakAnswer()">
            🔊 <span style="color:#ffffff !important;">Listen to Answer / </span><span>جواب سنیں</span>
        </button>
        
        <script>
        function speakAnswer() {
            window.speechSynthesis.cancel();
            // safe_js_string already contains surrounding quotes from json.dumps()
            var msg = new SpeechSynthesisUtterance(__SAFE_ANSWER__);
            msg.lang = 'en-IN'; msg.rate = 0.9;
            window.speechSynthesis.speak(msg);
        }
        </script>
    </body>
    </html>
    '''.replace("__SAFE_ANSWER__", safe_js_string)
    
    components.html(html_template, height=55)


# -----------------------------------
# BRIGHT & VIBRANT VISUAL ENGINE
# -----------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    /* Radiant Visual Canvas Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important;
    }
    
    .block-container {
        max-width: 750px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Strict Text Color System Force Rules */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span, label {
        color: #ffffff !important;
    }
    
    body, p, span, div, label {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Forced Nastaleeq Font Framework */
    .urdu-text, [lang="ur"], .stAlert p, .custom-label span, div[data-baseweb="input"] input {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', 'Urdu Typesetting', 'Nastaliq', sans-serif !important;
    }
    
    div[data-baseweb="input"] input {
        font-size: 22px !important;
        line-height: 1.8 !important;
        direction: auto !important;
    }
    
    .stAlert {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-left: 5px solid #00e676 !important;
        border-radius: 14px;
        backdrop-filter: blur(10px);
    }
    div.stTextInput > div > div > input {
        border-radius: 14px; 
        border: 2px solid #1a73e8; 
        padding: 14px; 
        background-color: #ffffff !important;
        color: #1a252f !important;
    }
    .answer-box {
        background-color: #ffffff !important; padding: 26px; border-radius: 18px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); border-left: 6px solid #00e676; margin: 20px 0;
    }
    .answer-title { color: #1e3c72 !important; font-weight: 700; font-size: 19px; }
    .answer-text { font-size: 16px; color: #1f2937 !important; line-height: 1.65; }
    
    /* QUANTUM CORE INTELLIGENCE SPHERE ANIMATION */
    .avatar-container {
        position: relative;
        width: 100px;
        height: 100px;
        margin: 20px auto 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .quantum-sphere {
        position: relative;
        width: 46px;
        height: 46px;
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #00e676 50%, #004d40 100%);
        border-radius: 50%;
        box-shadow: 0 0 25px rgba(0, 230, 118, 0.8), 0 0 45px rgba(30, 82, 152, 0.6);
        animation: sphereGlow 3s infinite ease-in-out;
        overflow: hidden;
    }
    
    /* Energy Contour rings running along the surface */
    .quantum-sphere::before {
        content: "";
        position: absolute;
        top: -10%; left: -10%; width: 120%; height: 120%;
        border: 3px double rgba(255, 255, 255, 0.35);
        border-radius: 45%;
        animation: surfaceWarp 4s infinite linear;
    }

    /* Outward ambient aura pulse (The Orbit) */
    .quantum-pulse-ring {
        position: absolute;
        width: 70px;
        height: 70px;
        border: 2px dashed rgba(0, 230, 118, 0.5);
        border-radius: 50%;
        animation: radarPulse 2.5s infinite cubic-bezier(0.215, 0.610, 0.355, 1);
        z-index: -1;
    }
    
    @keyframes sphereGlow {
        0%, 100% { transform: scale(1); box-shadow: 0 0 25px rgba(0, 230, 118, 0.8); }
        50% { transform: scale(1.06); box-shadow: 0 0 35px rgba(255,255,255,0.9), 0 0 50px rgba(0,230,118,1); }
    }
    
    @keyframes surfaceWarp {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes radarPulse {
        0% { transform: scale(0.6); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: scale(1.3); opacity: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# GEOMETRICALLY CENTERED BRANDING HEADER
# -----------------------------------

st.markdown(
'''
<div style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 12px;">
    <img src="https://images.seeklogo.com/logo-png/22/1/maulana-azad-national-urdu-university-logo-png_seeklogo-226045.png" 
         onerror="this.src='manuu_logo.png';" 
         style="width: 120px; height: auto;" />
</div>

<div style='text-align: center; margin-top: 5px; margin-bottom: 20px;'>

<div style='font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 19px; margin-bottom: 8px; letter-spacing: 0.5px; line-height: 1.3;'>
    <span style="color: #ffffff !important;">MAULANA AZAD NATIONAL URDU UNIVERSITY</span>
</div>

<div style='font-family: "Helvetica Neue", Arial, sans-serif; font-size: 15px; font-weight: 500; letter-spacing: 0.5px; margin-top: 0; margin-bottom: 10px; line-height: 1.4;'>
    <span style='font-size: 23px; font-weight: 800; color: #ffffff !important;'>C</span><span style="color: #ffffff !important;">entre for</span> 
    <span style='font-size: 23px; font-weight: 800; color: #ffffff !important;'>D</span><span style="color: #ffffff !important;">istance and</span> 
    <span style='font-size: 23px; font-weight: 800; color: #ffffff !important;'>O</span><span style="color: #ffffff !important;">nline</span> 
    <span style='font-size: 23px; font-weight: 800; color: #ffffff !important;'>E</span><span style="color: #ffffff !important;">ducation</span>
</div>

<div class="avatar-container">
    <div class="quantum-sphere"></div>
    <div class="quantum-pulse-ring"></div>
</div>

<div style="margin-bottom: 5px;">
<div style="font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', sans-serif !important; font-weight: 500; font-size: 52px; color: #00e676; line-height: 1.1; direction: rtl;" lang="ur">
معاوِن
</div>
<div style='font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 900; font-size: 40px; margin-top: -5px; margin-bottom: 2px; letter-spacing: 1px;'>
    <span style="color: #ffffff !important;">MAVIN</span>
</div>
</div>
<div style='font-size: 14px; margin-top: 2px; font-weight: 500; letter-spacing: 0.5px;'>
    <span style="color: rgba(255,255,255,0.95) !important;">(MANUU Virtual Interface)</span>
</div>
<div style='height: 3px; width: 140px; background: linear-gradient(90deg, #ffffff, #00e676); margin: 12px auto; border-radius: 2px;'></div>
</div>
''',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="stAlert" style="padding: 12px; margin-bottom: 15px;">
        <span style="color:#ffffff !important;">💡</span> <strong style="color:#ffffff !important;">Language Support:</strong> <span style="color:#ffffff !important;">You can type your questions comfortably in English or</span> <span class="urdu-text" style="font-family:'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', sans-serif !important; font-size:22px; color:#00e676; vertical-align:middle;">Urdu (اردو)</span>
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
    <label class="custom-label" style="font-size: 15px; font-weight: 500; margin-bottom: 6px; display: block;">
        <span style="color: #ffffff !important;">How can MAVIN assist you today? / </span>
        <span style="font-family:'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', 'Urdu Typesetting', sans-serif !important; font-size:24px; color:#00e676; vertical-align: middle; direction: rtl;">
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
            '''
            <div class="answer-box">
                <div class="answer-title">🤖 MAVIN:</div>
                <div class="answer-text">{}</div>
            </div>
            '''.format(result["answer"]),
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
