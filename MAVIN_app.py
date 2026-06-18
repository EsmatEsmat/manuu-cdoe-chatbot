# -*- coding: utf-8 -*-
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
st.set_page_config(page_title="MAVIN - CDOE MANUU", page_icon="manuu_logo.png", layout="centered")

if "messages" not in st.session_state:
    greeting = "Adaab! Welcome to MAVIN, your official assistant for MANUU CDOE. How can I help you today?\n\nآداب! مانو (MANUU) CDOE کے معاون میں آپ کا خیرمقدم ہے۔ میں آپ کی کیا مدد کر سکتا ہوں؟"
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

# -----------------------------------
# LOAD MODEL & DATA
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_faq():
    return pd.read_csv("master_faq.csv", encoding="latin1")

model = load_model()
faq = load_faq()

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
# BACKEND ENGINE
# -----------------------------------
def handle_keyword_overrides(cleaned_question):
    tokens = cleaned_question.split()
    for keyword in ["mba", "bca", "bcom", "bed", "ma", "ba"]:
        if keyword in tokens:
            matches = faq[faq["Main Question"].str.lower().str.contains(f"about {keyword}", na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                return {"answer": row["Answer"], "matched_question": row["Main Question"], "category": row.get("Category", "-"), "intent": row.get("Intent", "-"), "score": 1.0}
    return None

def get_answer(user_question):
    cleaned = clean_text(user_question)
    greetings = ["salam", "adaab", "hi", "hello", "hey", "namaste"]
    if any(g in cleaned for g in greetings):
        return {"answer": "Adaab! How can I help you?", "matched_question": "Greeting", "category": "General", "intent": "Greeting", "score": 1.0}
    
    direct = handle_keyword_overrides(cleaned)
    if direct: return direct

    user_emb = model.encode(cleaned, convert_to_tensor=True)
    val_m, idx_m = util.cos_sim(user_emb, faq_embs_main).max(dim=1)
    val_a, idx_a = util.cos_sim(user_emb, faq_embs_alt).max(dim=1)
    val_r, idx_r = util.cos_sim(user_emb, faq_embs_real).max(dim=1)
    
    best_score, best_idx = max([(val_m, idx_m), (val_a, idx_a), (val_r, idx_r)], key=lambda x: x[0])
    row = faq.iloc[int(best_idx)]
    
    if float(best_score) < 0.50:
        return {"answer": "I am sorry, I can only answer MANUU CDOE related questions.", "matched_question": "-", "category": "-", "intent": "-", "score": round(float(best_score), 3)}
    
    return {"answer": row["Answer"], "matched_question": row["Main Question"], "category": row.get("Category", "-"), "intent": row.get("Intent", "-"), "score": round(float(best_score), 3)}

def save_log(q, res, orig=""):
    log_data = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "User Query": q, "Bot Answer": res["answer"], "Score": res["score"]}
    pd.DataFrame([log_data]).to_csv("chat_logs.csv", mode="a", header=not os.path.exists("chat_logs.csv"), index=False, encoding="utf-8-sig")

def show_speech_button(text):
    safe_js = json.dumps(text)
    components.html(f'''<button onclick="window.speechSynthesis.speak(new SpeechSynthesisUtterance({safe_js}))" style="background:#00e676;border:none;padding:10px;border-radius:8px;cursor:pointer;">🔊 Listen / سنیں</button>''', height=50)

# -----------------------------------
# UI & CHAT LOOP
# -----------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("MAVIN is thinking..."):
            urdu = is_urdu(prompt)
            q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
            res = get_answer(q)
            ans = GoogleTranslator(source='en', target='ur').translate(res["answer"]) if urdu else res["answer"]
            
            st.markdown(ans)
            show_speech_button(ans)
            save_log(prompt, res)
            st.session_state.messages.append({"role": "assistant", "content": ans})
