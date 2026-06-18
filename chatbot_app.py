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
from deep_translator import GoogleTranslator

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(page_title="MAVIN - CDOE MANUU", page_icon="manuu_logo.png", layout="centered")

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

def is_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def get_answer(q):
    user_emb = model.encode(q, convert_to_tensor=True)
    val_m, idx_m = util.cos_sim(user_emb, faq_embs_main).max(dim=1)
    val_a, idx_a = util.cos_sim(user_emb, faq_embs_alt).max(dim=1)
    val_r, idx_r = util.cos_sim(user_emb, faq_embs_real).max(dim=1)
    best_score, best_idx = max([(val_m, idx_m), (val_a, idx_a), (val_r, idx_r)], key=lambda x: x[0])
    row = faq.iloc[int(best_idx)]
    return {"answer": row["Answer"], "score": float(best_score)}

# -----------------------------------
# FULL ORIGINAL VISUAL ENGINE
# -----------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important; }
    h1, h2, h3, p, div, span, label { color: #ffffff !important; }
    .quantum-sphere { width: 46px; height: 46px; background: radial-gradient(circle, #ffffff, #00e676); border-radius: 50%; box-shadow: 0 0 25px #00e676; margin: 20px auto; }
</style>
""", unsafe_allow_html=True)

# YOUR ORIGINAL BRANDING HEADER
st.markdown('''
<div style="text-align: center;">
    <img src="https://images.seeklogo.com/logo-png/22/1/maulana-azad-national-urdu-university-logo-png_seeklogo-226045.png" style="width: 120px;"/>
    <div style="font-weight: 700; font-size: 19px; margin-top: 10px;">MAULANA AZAD NATIONAL URDU UNIVERSITY</div>
    <div style="font-size: 15px; margin-top: 5px;">
        <span style="font-size: 23px; font-weight: 800;">C</span>entre for 
        <span style="font-size: 23px; font-weight: 800;">D</span>istance and 
        <span style="font-size: 23px; font-weight: 800;">O</span>nline 
        <span style="font-size: 23px; font-weight: 800;">E</span>ducation
    </div>
    <div class="quantum-sphere"></div>
    <div style="font-weight: 900; font-size: 40px;">MAVIN</div>
</div>
''', unsafe_allow_html=True)

# -----------------------------------
# CHAT INTERFACE
# -----------------------------------
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("MAVIN is thinking..."):
            urdu = is_urdu(prompt)
            q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
            res = get_answer(q)
            ans = GoogleTranslator(source='en', target='ur').translate(res["answer"]) if urdu else res["answer"]
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
