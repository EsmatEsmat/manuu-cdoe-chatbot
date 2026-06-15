
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


# -----------------------------------
# PAGE SETTINGS
# -----------------------------------

st.set_page_config(
    page_title="CDOE MANUU Chatbot",
    page_icon="cdoe_logo.png",  # Changing the graduation hat to your official logo file!
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
# CLEAN TEXT
# -----------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------
# LOAD MODEL + DATA
# -----------------------------------

model = load_model()
faq = load_faq()

faq_embeddings = model.encode(
    faq["search_text"].tolist(),
    convert_to_tensor=False
)


# -----------------------------------
# GET BEST ANSWER
# -----------------------------------

def get_answer(user_question):

    cleaned_question = clean_text(user_question)

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
        fuzzy_score = fuzz.WRatio(cleaned_question, row_text) / 100

        # FIX: Rely completely on the smart AI score for choosing the answer
        final_score = semantic_score

        combined_scores.append(final_score)

    best_index = max(
        range(len(combined_scores)),
        key=combined_scores.__getitem__
    )

    best_score = combined_scores[best_index]
    row = faq.iloc[best_index]

    # STAGE 2 FIX: Use a combination score JUST for the fallback safety net
    # This keeps your flight-to-Delhi question blocked perfectly!
    safety_check_score = (0.70 * best_score) + (0.30 * (fuzz.WRatio(cleaned_question, row["search_text"]) / 100))

    if safety_check_score < 0.45:
        return {
            "answer": "I am sorry, I can only answer MANUU CDOE related questions.",
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

def save_log(user_query, result):

    log_file = "chat_logs.csv"

    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Query": user_query,
        "Bot Answer": result["answer"],
        "Matched Question": result["matched_question"],
        "Category": result["category"],
        "Intent": result["intent"],
        "Confidence Score": result["score"]
    }

    log_df = pd.DataFrame([log_data])

    if os.path.exists(log_file):
        log_df.to_csv(
            log_file,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        log_df.to_csv(
            log_file,
            index=False,
            encoding="utf-8-sig"
        )


# -----------------------------------
# SPEECH BUTTON
# -----------------------------------

def show_speech_button(answer_text):

    safe_answer = (
        answer_text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
    )

    components.html(
        f"""
        <button onclick="speakAnswer()" style="
            background-color:#0f6b4f;
            color:white;
            border:none;
            padding:10px 16px;
            border-radius:8px;
            cursor:pointer;
            font-size:16px;">
            🔊 Listen to Answer
        </button>

        <script>
        function speakAnswer() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{safe_answer}');
            msg.lang = 'en-IN';
            msg.rate = 0.9;
            msg.pitch = 1;
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """,
        height=70
    )


# -----------------------------------
# STREAMLIT USER INTERFACE (UPGRADED)
# -----------------------------------

# Custom Premium University Dashboard Styling
st.markdown(
    """
    <style>
    /* Main Background: Warm vintage parchment / old paper feel */
    .stApp {
        background: linear-gradient(135deg, #f7f4eb 0%, #f1ede2 100%);
    }
    
    /* Center the main content area beautifully on widescreen monitors */
    .block-container {
        max-width: 800px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Info Banner Restyling: Muted vintage tint */
    .stAlert {
        background-color: #eae5d8 !important;
        border-left: 5px solid #16a34a !important;
        color: #27422b !important;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.01);
    }
    
    /* Input Box: Highlighting in clean Red when clicked */
    div.stTextInput > div > div > input {
        border-radius: 14px;
        border: 2px solid #d3cbba;
        padding: 14px;
        font-size: 16px;
        background-color: #ffffff;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
    }
    div.stTextInput > div > div > input:focus {
        border-color: #dc2626 !important;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.15) !important;
    }
    
    /* Chat Answer Card: Super clean card that pops out against the parchment background */
    .answer-box {
        background-color: #ffffff;
        padding: 26px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04);
        border: 1px solid #e1dacb;
        border-left: 6px solid #16a34a;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .answer-title {
        color: #15803d;
        font-weight: 700;
        font-size: 19px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .answer-text {
        font-size: 16.5px;
        color: #1f2937;
        line-height: 1.65;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Centered Logo Section
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    try:
        st.image("cdoe_logo.png", width=160)
    except:
        pass

# Beautiful Institutional Title Styling
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 30px;'>
        <h2 style='color: #dc2626; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 25px; margin-bottom: 4px; letter-spacing: 0.5px;'>
            MAULANA AZAD NATIONAL URDU UNIVERSITY
        </h2>
        <p style='color: #cca43b; font-size: 24px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0; margin-bottom: 10px;'>
            Centre for Distance & Online Education
        </p>
        <h1 style='color: #1f2937; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 800; font-size: 30px; margin-top: 0; margin-bottom: 5px;'>
            🎓 CDOE Student Support
        </h1>
        <div style='height: 3px; width: 160px; background-color: #cca43b; margin: 15px auto; border-radius: 2px;'></div>
    </div>
    """,
    unsafe_allow_html=True
)
# User Query Interaction
st.info("💡 **Voice Search Tip:** Click inside the box below and press **Windows Key + H** to speak your question aloud!")

student_query = st.text_input(
    "How can I help you today?",
    placeholder="Type your question here (e.g., MBA eligibility criteria)..."
)

if student_query:
    with st.spinner("✨ Assistant is typing..."):
        result = get_answer(student_query)
        save_log(student_query, result)

    # Display Answer inside custom beautiful happy box
    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-title">✨ Response for You</div>
            <div class="answer-text">{result["answer"]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Audio playback button
    show_speech_button(result["answer"])

    # Match Details inside clean dropdown
    with st.expander("📊 Technical Analytics (For Office Evaluation)"):
        st.markdown(f"**Confidence Match Score:** `{result['score']}`")
        st.markdown(f"**Mapped Database Intent:** `{result['intent']}`")
        st.markdown(f"**Category:** `{result['category']}`")
        st.markdown(f"**Reference Master Question:** *\"{result['matched_question']}\"*")
