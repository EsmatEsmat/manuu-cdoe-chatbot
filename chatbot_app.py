# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:04:59 2026

@author: ismat
"""
import ssl
# Tell Python to ignore the broken Windows certificate store entirely
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

# LIVE TRANSLATION ENGINE INTEGRATION
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
# LOAD AI MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------
# LOAD FAQ DATA
# -----------------------------------

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


# -----------------------------------
# CLEAN TEXT UTILITIES
# -----------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_urdu(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))


# -----------------------------------
# INITIALIZE MODEL + DATA
# -----------------------------------

model = load_model()
faq = load_faq()

faq_embeddings = model.encode(
    faq["search_text"].tolist(),
    convert_to_tensor=False
)


# -----------------------------------
# DETECT SHORT KEYWORD DIRECT MATCHES
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


# -----------------------------------
# GET BEST ANSWER
# -----------------------------------

def get_answer(user_question):
    cleaned_question = clean_text(user_question)

    direct_match = handle_keyword_overrides(cleaned_question)
    if direct_match:
        return direct_match

    question_embedding = model.encode(
        [cleaned_question],
        convert_to_tensor=False
    )

    semantic_scores = cosine_similarity(
        question_embedding,
        faq_embeddings
    )[0]

    combined_scores = []
    for i, row_text in enumerate(faq["search_text"]):
        semantic_score = float(semantic_scores[i])
        combined_scores.append(semantic_score)

    best_index = max(
        range(len(combined_scores)),
        key=combined_scores.__getitem__
    )

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


# -----------------------------------
# SAVE CHAT LOG
# -----------------------------------

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
# SPEECH ENGINE INTERFACE (BUG-FREE RESYNC)
# -----------------------------------

def show_speech_button(answer_text):
    safe_answer = (
        answer_text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
    )

    # Uses a zero-margin wrapper script to permanently kill the white ghosting container box bug
    components.html(
        f"""
        <div style="margin:0; padding:0; background:transparent; overflow:hidden;">
            <button onclick="speakAnswer()" style="
                background-color:#00c853;
                color:white;
                border:none;
                padding:10px 16px;
                border-radius:8px;
                cursor:pointer;
                font-weight:600;
                font-size:15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                🔊 Listen to Answer / جواب سنیں
            </button>
        </div>

        <script>
        function speakAnswer() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{safe_answer}');
            msg.lang = 'en-IN';
            msg.rate = 0.9;
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """,
        height=50
    )


# -----------------------------------
# STREAMLIT USER INTERFACE FRAMEWORK
# -----------------------------------

st.markdown(
    """
    <style>
    /* Vibrant presentation gradient shifting from Dark Sky Blue to Bright Green */
    .stApp {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 50%, #00c853 100%) !important;
    }
    .stMarkdown, p, h1, h2, h3, h4, span, label, li {
        color: #ffffff !important;
    }
    .block-container {
        max-width: 750px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Strict global layout configuration forcing Jameel Noori Nastaleeq for all Urdu strings */
    @import url('https://fonts.cdnfonts.com/css/jameel-noori-nastaleeq');
    
    body, *, .stAlert, div, input, p, span, label {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Look for any Urdu script block pattern on page and map to Nastaleeq layout */
    [lang="ur"], .urdu-text, .stAlert p, div.stTextInput label, div.stTextInput input::placeholder {
        font-family: 'Jameel Noori Nastaleeq', 'Urdu Typesetting', 'Nastaliq', sans-serif !important;
    }
    
    .stAlert {
        background-color: rgba(255, 255, 255, 0.12) !important;
        border-left: 5px solid #00c853 !important;
        color: #ffffff !important;
        border-radius: 14px;
        backdrop-filter: blur(8px);
    }
    div.stTextInput > div > div > input {
        border-radius: 14px;
        border: 2px solid #1a73e8;
        padding: 14px;
        font-size: 16px;
        background-color: #ffffff !important;
        color: #1a252f !important;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #00c853 !important;
    }
    .answer-box {
        background-color: #ffffff !important;
        padding: 26px;
        border-radius: 18px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        border-left: 6px solid #00c853;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .answer-title {
        color: #0d47a1 !important;
        font-weight: 700;
        font-size: 19px;
        margin-bottom: 10px;
    }
    .answer-text {
        font-size: 16px;
        color: #1f2937 !important;
        line-height: 1.65;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
    }
    
    /* PREMIUM GLOW-PULSE TITLE KEYFRAME ANIMATION */
    @keyframes titleGlowPulse {
        0% { text-shadow: 0 0 10px rgba(0,200,83,0.2); transform: scale(1); }
        50% { text-shadow: 0 0 25px rgba(0,200,83,0.8), 0 0 40px rgba(26,115,232,0.4); transform: scale(1.02); }
        100% { text-shadow: 0 0 10px rgba(0,200,83,0.2); transform: scale(1); }
    }
    .animated-title {
        display: inline-block;
        animation: titleGlowPulse 4s infinite ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App branding Header Row containing the official MANUU Crest
col1, col2, col3 = st.columns([1, 1.2, 1])
with col2:
    try:
        st.image("manuu_logo.png", width=140)
    except:
        pass

st.markdown(
"""
<div style='text-align: center; margin-top: 15px; margin-bottom: 25px;'>
<h2 style='color: #ffffff; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 600; font-size: 21px; margin-bottom: 6px; letter-spacing: 0.5px;'>
MAULANA AZAD NATIONAL URDU UNIVERSITY
</h2>
<p style='color: #f1c40f; font-size: 19px; font-weight: 500; letter-spacing: 0.5px; margin-top: 0; margin-bottom: 20px;'>
<span style='font-size: 26px; font-weight: 800;'>C</span>entre for 
<span style='font-size: 26px; font-weight: 800;'>D</span>istance & 
<span style='font-size: 26px; font-weight: 800;'>O</span>nline 
<span style='font-size: 26px; font-weight: 800;'>E</span>ducation
</p>
<div class="animated-title" style="margin-bottom: 10px;">
<div style="font-family: 'Jameel Noori Nastaleeq', 'Urdu Typesetting', sans-serif !important; font-weight: 500; font-size: 56px; color: #00c853; line-height: 1.1; direction: rtl;" lang="ur">
مَعاوِن
</div>
<div style='color: #ffffff; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 900; font-size: 44px; margin-top: -5px; margin-bottom: 2px; letter-spacing: 1px;'>
MAVIN
</div>
</div>
<p style='color: rgba(255,255,255,0.9); font-size: 16px; margin-top: 2px; font-weight: 500; letter-spacing: 0.5px;'>
(MANUU Virtual Interface)
</p>
<div style='height: 3px; width: 160px; background: linear-gradient(90deg, #f1c40f, #00c853); margin: 15px auto; border-radius: 2px;'></div>
</div>
""",
    unsafe_allow_html=True
)

# Custom container class structure injects Nastaleeq dynamically into the info panel string text blocks
st.markdown(
    """
    <div class="stAlert" style="padding: 15px; margin-bottom: 20px;">
        <span style="color:#f1c40f;">💡</span> <strong>Language Support:</strong> You can type your questions comfortably in English or <span style="font-family:'Jameel Noori Nastaleeq', sans-serif; font-size:20px; vertical-align:middle; color:#00c853;">Urdu (اردو)</span>!
    </div>
    """, 
    unsafe_allow_html=True
)

if "show_analytics" not in st.session_state:
    st.session_state.show_analytics = False

# Uses raw HTML generation targeting wrapper labels to preserve font styles across the text input bar
st.markdown('<label style="color: white; font-size: 16px; font-weight: 500; margin-bottom: 8px; display: block;">How can MAVIN assist you today? / <span style="font-family:\'Jameel Noori Nastaleeq\', sans-serif; font-size:22px; color:#00c853; vertical-align: middle;">میں آپ کی کیا مدد کر سکتا ہوں؟</span></label>', unsafe_allow_html=True)
student_query = st.text_input("", placeholder="Type here...", label_visibility="collapsed")

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
                    st.toast(f"Translated query: {processed_query}")
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
