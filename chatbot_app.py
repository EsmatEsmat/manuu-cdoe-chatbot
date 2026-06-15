
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
    page_icon="🎓",
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

# Custom Happy/Professional CSS Styling
st.markdown(
    """
    <style>
    /* Main Background and Text Settings */
    .stApp {
        background: linear-gradient(to bottom, #f4fbf7, #ffffff);
    }
    
    /* Global Info Banner Restyling */
    .stAlert {
        background-color: #eafaf1 !important;
        border-left: 5px solid #0f6b4f !important;
        color: #0c523d !important;
        border-radius: 10px;
    }
    
    /* Input Box Focus Border styling */
    div.stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #ced4da;
        padding: 12px;
        font-size: 16px;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #0f6b4f !important;
        box-shadow: 0 0 0 0.2rem rgba(15, 107, 79, 0.25) !important;
    }
    
    /* Chat Answer Box Styling */
    .answer-box {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e1e8e5;
        border-left: 6px solid #0f6b4f;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .answer-title {
        color: #0f6b4f;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .answer-text {
        font-size: 16px;
        color: #2d3748;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered Logo Implementation
try:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
            <img st.image("cdoe_logo.png", width=160) style="object-fit: contain;">
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    pass

# Beautiful Institutional Title Styling
st.markdown(
    """
    <div style='text-align: center; margin-top: -10px; margin-bottom: 25px;'>
        <h2 style='color: #0f6b4f; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 700; font-size: 24px; margin-bottom: 3px; letter-spacing: 0.5px;'>
            MAULANA AZAD NATIONAL URDU UNIVERSITY
        </h2>
        <h1 style='color: #2d3748; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 800; font-size: 30px; margin-top: 0; margin-bottom: 5px;'>
            🎓 CDOE Support Bot
        </h1>
        <p style='color: #cca43b; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0;'>
            Centre for Distance & Online Education
        </p>
        <div style='height: 3px; width: 120px; background-color: #cca43b; margin: 15px auto; border-radius: 2px;'></div>
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
