# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:04:59 2026

@author: ismat
"""
import ssl
ssl.create_default_context = lambda *args, **kwargs: ssl._create_unverified_context(*args, **kwargs)
import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

st.set_page_config(page_title="MAVIN - CDOE MANUU", page_icon="manuu_logo.png", layout="centered")

# -----------------------------------
# FULL CSS BLOCK - FIXED ALIGNMENT
# -----------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Jameel+Noori+Nastaleeq&display=swap');

    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important; }
    h1, h2, h3, p, div, span, label { color: #ffffff !important; }
    
    /* Apply Nastaleeq font and RTL specifically to containers containing Urdu */
    .stChatMessage, .stChatInput textarea, div[data-testid="stMarkdown"] p {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif !important;
        font-size: 22px !important;
    }
    
    /* Logic: If text contains Urdu characters, force RTL alignment */
    .stChatMessage[data-testid="stChatMessage"]:has(p[lang="ur"]),
    .stChatMessage[data-testid="stChatMessage"]:has(p:lang(ur)) {
        direction: rtl !important;
        text-align: right !important;
    }

    .urdu-title { font-family: 'Jameel Noori Nastaleeq', serif !important; font-size: 52px !important; color: #00e676 !important; }

    /* QUANTUM CORE & DOTTED PULSE AURA */
    .avatar-container { position: relative; width: 120px; height: 120px; margin: 20px auto; display: flex; align-items: center; justify-content: center; }
    .quantum-sphere { position: relative; width: 46px; height: 46px; background: radial-gradient(circle at 30% 30%, #ffffff 0%, #00e676 50%, #004d40 100%); border-radius: 50%; box-shadow: 0 0 25px rgba(0, 230, 118, 0.8); animation: sphereGlow 3s infinite ease-in-out; z-index: 2; }
    .quantum-pulse-ring { position: absolute; width: 70px; height: 70px; border: 2px dashed #00e676; border-radius: 50%; animation: radarPulse 2.5s infinite cubic-bezier(0.215, 0.610, 0.355, 1); z-index: 1; }
    
    @keyframes sphereGlow { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.06); } }
    @keyframes radarPulse { 0% { transform: scale(0.6); opacity: 0; } 50% { opacity: 1; } 100% { transform: scale(1.3); opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# BRANDING HEADER
# -----------------------------------
st.markdown('''
<div style="text-align: center;">
    <img src="https://images.seeklogo.com/logo-png/22/1/maulana-azad-national-urdu-university-logo-png_seeklogo-226045.png" style="width: 120px;"/>
    <div style="font-weight: 700; font-size: 19px; margin-top: 10px;">MAULANA AZAD NATIONAL URDU UNIVERSITY</div>
    <div style="font-size: 15px; margin-top: 5px;">
        <span style="font-size: 23px; font-weight: 800;">C</span>entre for <span style="font-size: 23px; font-weight: 800;">D</span>istance and <span style="font-size: 23px; font-weight: 800;">O</span>nline <span style="font-size: 23px; font-weight: 800;">E</span>ducation
    </div>
    <div class="avatar-container">
        <div class="quantum-sphere"></div>
        <div class="quantum-pulse-ring"></div>
    </div>
    <div class="urdu-title">معاوِن</div>
    <div style="font-weight: 900; font-size: 40px; margin-top: -10px;">MAVIN</div>
    <div style="font-size: 14px; margin-top: 2px;">(MANUU Virtual Interface)</div>
</div>
''', unsafe_allow_html=True)

# -----------------------------------
# LOGIC
# -----------------------------------
@st.cache_resource
def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_data
def load_faq(): return pd.read_csv("master_faq.csv", encoding="latin1")

model, faq = load_model(), load_faq()
faq_embs_main = model.encode(faq["Main Question"].fillna("").tolist(), convert_to_tensor=True)

def is_urdu(text): return bool(re.search(r'[\u0600-\u06FF]', text))

if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        cleaned = prompt.lower().strip()
        urdu = is_urdu(prompt)
        if any(greet in cleaned for greet in ["hi", "hello", "salam", "adaab", "hey"]):
            ans = "Adaab! Welcome to MAVIN." if not urdu else "آداب! مانو CDOE کے معاون میں خوش آمدید۔"
        else:
            q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
            val, idx = util.cos_sim(model.encode(q, convert_to_tensor=True), faq_embs_main).max(dim=1)
            ans = faq.iloc[int(idx)]["Answer"] if val > 0.4 else "I can only answer MANUU CDOE related questions."
            if urdu: ans = GoogleTranslator(source='en', target='ur').translate(ans)
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
