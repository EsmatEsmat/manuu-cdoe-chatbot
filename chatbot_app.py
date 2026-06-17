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
    # FIXED: Repaired the broken pd. statement into a full DataFrame initialization
    log_df = pd.DataFrame([log_data])
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        log_df.to_csv(log_file, index=False, encoding="utf-8-sig")


# -----------------------------------
# HARD-FORCED IFRAME AUDIO ENGINE
# -----------------------------------
def show_speech_button(answer_text):
    safe_answer = answer_text.replace(r"\\", r"\\\\").replace(r"'", r"\'").replace(r'"', r'\"').replace("\n", " ")
    
    html_template = """
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
            var msg = new SpeechSynthesisUtterance('__SAFE_ANSWER__');
            msg.lang = 'en-IN'; msg.rate = 0.9;
            window.speechSynthesis.speak(msg);
        }
        </script>
    </body>
    </html>
    """.replace("__SAFE_ANSWER__", safe_answer)
    
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
        background: radial-gradient
