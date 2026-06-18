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
from sentence_transformers import SentenceTransformer, util
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
    return pd.read_csv("master_faq.csv", encoding="latin1")

model = load_model()
faq = load_faq()

# Pre-calculate embeddings
faq_embs_main = model.encode(faq["Main Question"].fillna("").tolist(), convert_to_tensor=True)
faq_embs_alt = model.encode(faq["Alternate Questions"].fillna("").tolist(), convert_to_tensor=True)
faq_embs_real = model.encode(faq["Real Student Variants"].fillna("").tolist(), convert_to_tensor=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

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
    greetings = ["salam", "adaab", "hi", "hello", "hey", "namaste", "assalam"]
    if any(greet in cleaned_question for greet in greetings):
        return {
            "answer": "Adaab! Welcome to MAVIN, your official assistant for MANUU CDOE. How can I help you today?\n\nآداب! مانو (MANUU) CDOE کے معاون میں آپ کا خیرمقدم ہے۔ میں آپ کی کیا مدد کر سکتا ہوں؟",
            "matched_question": "Greeting",
            "category": "General",
            "intent": "Greeting",
            "score": 1.0
        }

    direct_match = handle_keyword_overrides(cleaned_question)
    if direct_match: return direct_match

    user_emb = model.encode(cleaned_question, convert_to_tensor=True)
    val_m, idx_m = util.cos_sim(user_emb, faq_embs_main).max(dim=1)
    val_a, idx_a = util.cos_sim(user_emb, faq_embs_alt).max(dim=1)
    val_r, idx_r = util.cos_sim(user_emb, faq_embs_real).max(dim=1)
    
    best_score, best_idx = max([(val_m, idx_m), (val_a, idx_a), (val_r, idx_r)], key=lambda x: x[0])
    row = faq.iloc[int(best_idx)]

    if float(best_score) < 0.50: 
        return {"answer": "I am sorry, I can only answer MANUU CDOE related questions.", "matched_question": "No match", "category": "-", "intent": "-", "score": round(float(best_score), 3)}

    return {"answer": row["Answer"], "matched_question": row["Main Question"], "category": row.get("Category", "-"), "intent": row.get("Intent", "-"), "score": round(float(best_score), 3)}

def save_log(user_query, result, original_urdu=""):
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Query": user_query if not original_urdu else f"{original_urdu} (Translated: {user_query})",
        "Bot Answer": result["answer"],
        "Matched Question": result["matched_question"],
        "Category": result["category"],
        "Intent": result["intent"],
        "Confidence Score": result["score"]
    }
    pd.DataFrame([log_data]).to_csv("chat_logs.csv", mode="a", header=not os.path.exists("chat_logs.csv"), index=False, encoding="utf-8-sig")

def show_speech_button(answer_text):
    safe_js_string = json.dumps(answer_text)
    html_template = f'''
    <button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_js_string}))" style="background:#00e676;border:none;padding:10px;border-radius:8px;cursor:pointer;color:white;font-weight:bold;">
        🔊 Listen / جواب سنیں
    </button>'''
    components.html(html_template, height=50)

# -----------------------------------
# VISUAL ENGINE (CSS)
# -----------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important; }
    .avatar-container { margin: 20px auto; display: flex; justify-content: center; }
    .quantum-sphere { width: 46px; height: 46px; background: radial-gradient(circle, #ffffff, #00e676); border-radius: 50%; box-shadow: 0 0 25px #00e676; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# UI / CHAT
# -----------------------------------
st.markdown("<h1 style='text-align: center; color: white;'>MAVIN - CDOE MANUU</h1>", unsafe_allow_html=True)
st.markdown("<div class='avatar-container'><div class='quantum-sphere'></div></div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Adaab! Welcome to MAVIN, your official assistant for MANUU CDOE. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("✨ MAVIN is thinking..."):
            urdu_detected = is_urdu(prompt)
            processed_query = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu_detected else prompt
            result = get_answer(processed_query)
            final_answer = GoogleTranslator(source='en', target='ur').translate(result["answer"]) if urdu_detected else result["answer"]
            
            st.markdown(final_answer)
            show_speech_button(final_answer)
            save_log(processed_query, result, original_urdu=prompt if urdu_detected else "")
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
