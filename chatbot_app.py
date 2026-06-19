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

# --- 1. CONFIGURATION & AUTH ---
st.set_page_config(page_title="MAVIN - CDOE MANUU", page_icon="manuu_logo.png", layout="centered")

def check_password():
    def password_entered():
        if st.session_state["password"] == "MAVINadmin2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.sidebar.text_input("Admin Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.sidebar.text_input("Admin Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    return True

# --- 2. CSS & DYNAMIC ALIGNMENT SCRIPT ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Jameel+Noori+Nastaleeq&display=swap');
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 40%, #00e676 100%) !important; }
    h1, h2, h3, p, div, span, label { color: #ffffff !important; }
    .stChatMessage, .stChatInput textarea, div[data-testid="stMarkdown"] p {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif !important;
        font-size: 22px !important;
    }
    .urdu-title { font-family: 'Jameel Noori Nastaleeq', serif !important; font-size: 52px !important; color: #00e676 !important; }
    .avatar-container { position: relative; width: 120px; height: 120px; margin: 20px auto; display: flex; align-items: center; justify-content: center; }
    .quantum-sphere { position: relative; width: 46px; height: 46px; background: radial-gradient(circle at 30% 30%, #ffffff 0%, #00e676 50%, #004d40 100%); border-radius: 50%; box-shadow: 0 0 25px rgba(0, 230, 118, 0.8); animation: sphereGlow 3s infinite ease-in-out; z-index: 2; }
    .quantum-pulse-ring { position: absolute; width: 70px; height: 70px; border: 2px dashed #00e676; border-radius: 50%; animation: radarPulse 2.5s infinite cubic-bezier(0.215, 0.610, 0.355, 1); z-index: 1; }
    @keyframes sphereGlow { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.06); } }
    @keyframes radarPulse { 0% { transform: scale(0.6); opacity: 0; } 50% { opacity: 1; } 100% { transform: scale(1.3); opacity: 0; } }
</style>

<script>
    const observer = new MutationObserver(() => {
        document.querySelectorAll('[data-testid="stChatMessage"]').forEach((msg) => {
            if (/[\u0600-\u06FF]/.test(msg.innerText)) {
                msg.style.direction = 'rtl'; msg.style.textAlign = 'right';
            } else {
                msg.style.direction = 'ltr'; msg.style.textAlign = 'left';
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# --- 3. BACKEND ---
@st.cache_resource
def load_model(): return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_data
def load_faq(): return pd.read_csv("master_faq.csv", encoding="latin1")

model, faq = load_model(), load_faq()
faq_embs_main = model.encode(faq["Main Question"].fillna("").tolist(), convert_to_tensor=True)
def is_urdu(text): return bool(re.search(r'[\u0600-\u06FF]', text))

# --- 4. SIDEBAR & LOGIC ---
with st.sidebar:
    st.header("⚙️ Admin Settings")
    if check_password():
        st.success("Admin Access Granted")
        if st.button("Logout"):
            del st.session_state["password_correct"]
            st.rerun()
        action = st.radio("Action", ["Chat", "Edit FAQ"])
    else:
        action = "Chat"

if action == "Edit FAQ":
    st.header("📝 Edit Master FAQ")
    edited_df = st.data_editor(faq, num_rows="dynamic")
    if st.button("Save Changes"):
        edited_df.to_csv("master_faq.csv", index=False, encoding="latin1")
        st.success("FAQ Saved! Refresh to apply.")
else:
    st.markdown('''<div style="text-align: center;">
        <img src="https://images.seeklogo.com/logo-png/22/1/maulana-azad-national-urdu-university-logo-png_seeklogo-226045.png" style="width: 120px;"/>
        <div style="font-weight: 700; font-size: 19px; margin-top: 10px;">MAULANA AZAD NATIONAL URDU UNIVERSITY</div>
        <div class="avatar-container"><div class="quantum-sphere"></div><div class="quantum-pulse-ring"></div></div>
        <div class="urdu-title">معاوِن</div>
        <div style="font-weight: 900; font-size: 40px; margin-top: -10px;">MAVIN</div>
    </div>''', unsafe_allow_html=True)

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            cleaned = prompt.lower().strip()
            urdu = is_urdu(prompt)
            if any(g in cleaned for g in ["hi", "hello", "salam", "adaab", "hey", "سلام", "آداب"]):
                ans = "آداب! مانو CDOE کے معاون میں خوش آمدید۔" if urdu else "Adaab! Welcome to MAVIN."
            else:
                q = GoogleTranslator(source='ur', target='en').translate(prompt) if urdu else prompt
                val, idx = util.cos_sim(model.encode(q, convert_to_tensor=True), faq_embs_main).max(dim=1)
                ans = faq.iloc[int(idx)]["Answer"] if val > 0.4 else ("معذرت، میں صرف مانو CDOE سے متعلق سوالات دے سکتا ہوں۔" if urdu else "I can only answer MANUU CDOE related questions.")
                if urdu and val > 0.4: ans = GoogleTranslator(source='en', target='ur').translate(ans)
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
