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
from deep_translator import GoogleTranslator

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(page_title="MAVIN - CDOE MANUU", page_icon="manuu_logo.png", layout="centered")

# -----------------------------------
# INITIALIZE STATE
# -----------------------------------
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

# Embeddings
faq_embs_main = model.encode(faq["Main Question"].fillna("").tolist(), convert_to_tensor=True)
faq_embs_alt = model.encode(faq["Alternate Questions"].fillna("").tolist(), convert_to_tensor=True)
faq_embs_real = model.encode(faq["Real Student Variants"].fillna("").tolist(), convert_to_tensor=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def is_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def get_answer(user_question):
    cleaned = clean_text(user_question)
    user_emb = model.encode(cleaned, convert_to_tensor=True)
    
    val_m, idx_m = util.cos_sim(user_emb, faq_embs_main).max(dim=1)
    val_a, idx_a = util.cos_sim(user_emb, faq_embs_alt).max(dim=1)
    val_r, idx_r = util.cos_sim(user_emb, faq_embs_real).max(dim=1)
    
    best_score, best_idx = max([(val_m, idx_m), (val_a, idx_a), (val_r, idx_r)], key=lambda x: x[0])
    row = faq.iloc[int(best_idx)]
    
    score = float(best_score)
    if score < 0.50:
        return {"answer": "I am sorry, I can only answer MANUU CDOE related questions.", "score": score}
    
    return {"answer": row["Answer"], "score": score}

def save_log(q, res):
    pd.DataFrame([{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Query": q, "Answer": res["answer"]}]) \
      .to_csv("chat_logs.csv", mode="a", header=not os.path.exists("chat_logs.csv"), index=False, encoding="utf-8-sig")

# -----------------------------------
# CSS / VISUALS
# -----------------------------------
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important; }
    .avatar-container { display: flex; justify-content: center; margin: 20px 0; }
    .quantum-sphere { width: 50px; height: 50px; background: white; border-radius: 50%; box-shadow: 0 0 20px #00e676; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# UI
# -----------------------------------
st.markdown("<h1 style='text-align: center; color: white;'>MAVIN - MANUU CDOE</h1>", unsafe_allow_html=True)
st.markdown("<div class='avatar-container'><div class='quantum-sphere'></div></div>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask MAVIN..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            urdu = is_urdu(prompt)
            q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
            res = get_answer(q)
            ans = GoogleTranslator(source='en', target='ur').translate(res["answer"]) if urdu else res["answer"]
            
            st.markdown(ans)
            save_log(prompt, res)
            st.session_state.messages.append({"role": "assistant", "content": ans})
