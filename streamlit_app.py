# app.py

import os
import json
import tempfile
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
from google.cloud import texttospeech

# ---------------------------------------------------------
# 0) Page Styling
# ---------------------------------------------------------
st.set_page_config(page_title="VARC Voice-Assistant MVP", layout="wide")
st.markdown(
    """
    <link rel="stylesheet" href="static/style.css">
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 1) Bootstrap Secrets for Google TTS & OpenAI Whisper
# ---------------------------------------------------------
# Write GCP service account JSON (from st.secrets) to a temp file:
sa_json = st.secrets["gcp_tts"]["service_account"]
tts_sa_path = "/tmp/gcp_tts.json"
with open(tts_sa_path, "w") as f:
    f.write(sa_json)
# Point Google client to that JSON:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tts_sa_path

# Load OpenAI key from st.secrets and assign to openai.api_key:
openai.api_key = st.secrets["openai"]["api_key"]

# ---------------------------------------------------------
# 2) Caching Models & Data Loading
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nlp_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp_model = load_nlp_model()

@st.cache_data(show_spinner=False)
def load_data():
    with open("data/passages.json", "r") as f:
        passages = json.load(f)
    with open("data/questions.json", "r") as f:
        questions = json.load(f)
    with open("data/paragraph_summaries.json", "r") as f:
        para_summaries = json.load(f)
    return passages, questions, para_summaries

passages, questions_data, paragraph_summaries = load_data()

# ---------------------------------------------------------
# 3) Utility Functions
# ---------------------------------------------------------
def embed_texts(text_list):
    return nlp_model.encode(text_list, convert_to_tensor=True)

def score_paraphrase(user_text, gold_text):
    if not user_text or not gold_text:
        return 0
    u_emb = nlp_model.encode(user_text, convert_to_tensor=True)
    g_emb = nlp_model.encode(gold_text, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(u_emb, g_emb).item()
    if sim > 0.75:
        return 10
    elif sim > 0.5:
        return 5
    else:
        return 0

# ---------------------------------------------------------
# 4) OpenAI Whisper ASR
# ---------------------------------------------------------
def transcribe_whisper(audio_bytes, model_name="whisper-1"):
    """
    Uploads audio_bytes (WAV/MP3) to OpenAI Whisper and returns the transcript.
    """
    try:
        # Save to a temporary file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name

        audio_file = open(tmp_path, "rb")
        transcript = openai.Audio.transcribe(model=model_name, file=audio_file)
        audio_file.close()
        os.remove(tmp_path)
        return transcript.get("text", "")
    except Exception as e:
        st.error(f"Whisper ASR Error: {e}")
        return ""

# ---------------------------------------------------------
# 5) Google TTS
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_tts_client():
    return texttospeech.TextToSpeechClient()

def synthesize_text_to_mp3(text, voice_name="en-US-Wavenet-D", speaking_rate=1.0):
    """
    Converts text to MP3 bytes via Google TTS.
    """
    client = get_tts_client()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )
    try:
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Google TTS Error: {e}")
        return b""

# ---------------------------------------------------------
# 6) Initialize Session State
# ---------------------------------------------------------
if "user_data" not in st.session_state:
    st.session_state.user_data = {
        "keyword_flags": {},            # {question_id: [keywords]}
        "underlined_paragraph": None,    # selected paragraph number
        "pivot_paragraphs": [],          # list of paragraph numbers flagged as pivot
        "paragraph_understandings": {},  # {para_num: text}
        "question_paragraph_match": {},  # {question_id: para_num}
        "inference_answers": {},         # {question_id: text}
        "vocab_paraphrases": {},         # {question_id: text}
        "question_paraphrases": {},      # {question_id: text}
        "para_asr_text": {},             # {para_num: asr transcript}
    }

# ---------------------------------------------------------
# 7) Streamlit UI
# ---------------------------------------------------------
st.title("📚 VARC Voice-Assistant MVP")
st.markdown(
    """
**Instructions (MVP):**  
1. Select a passage.  
2. Preview questions and type any keywords you’d flag.  
3. Read/listen to each paragraph → click “Underline as Thesis” or “Flag as Pivot.”  
4. For each paragraph, choose to speak your understanding (via file upload) or type it.  
5. Match each question to a paragraph.  
6. For inference/vocab questions, type your answer/paraphrase.  
7. Paraphrase each question in your own words.  
8. Submit to view a scoring summary.  
"""
)

# ---------------------------------------------------------
# 8) Sidebar: TTS & ASR Settings
# ---------------------------------------------------------
st.sidebar.header("Settings")
use_tts = st.sidebar.checkbox("Enable TTS Playback", value=True)
use_asr = st.sidebar.checkbox("Enable Whisper ASR (Upload Audio)", value=False)

# ---------------------------------------------------------
# 9) Step 0: Select Passage
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 0: Select a CAT VARC Passage")
passage_keys = list(passages.keys())
selected_passage_key = st.selectbox("Choose a passage:", passage_keys)
selected_passage = passages[selected_passage_key]
selected_questions = questions_data[selected_passage_key]
selected_summaries = paragraph_summaries[selected_passage_key]

st.markdown(f"**{selected_passage_key}: {selected_passage.get('title','')}**")

# ---------------------------------------------------------
# 10) Step 1: Preview Questions & Flag Keywords
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 1: Preview All Questions & Flag Keywords")

for q in selected_questions:
    qid = q["id"]
    st.markdown(f"**{qid}. {q['text']}**")
    kw_input = st.text_input(
        f"Type keywords you’d flag for {qid} (comma-separated):",
        key=f"kw_{qid}",
        placeholder="e.g., lack, training",
    )
    st.session_state.user_data["keyword_flags"][qid] = [
        w.strip().lower() for w in kw_input.split(",") if w.strip()
    ]

# ---------------------------------------------------------
# 11) Step 2: Read & Annotate Paragraphs
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 2: Read & Listen to Each Paragraph, Annotate & Explain Understanding")

for idx, para_text in selected_passage["paragraphs"].items():
    para_num = int(idx)
    st.markdown(f"**Paragraph {para_num}:** {para_text}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Underline as Thesis (Para {para_num})", key=f"btn_thesis_{para_num}"):
            st.session_state.user_data["underlined_paragraph"] = para_num
    with col2:
        if st.button(f"Flag as Pivot (Para {para_num})", key=f"btn_pivot_{para_num}"):
            st.session_state.user_data["pivot_paragraphs"].append(para_num)

    # TTS playback if enabled
    if use_tts:
        tts_bytes = synthesize_text_to_mp3(para_text)
        st.audio(tts_bytes, format="audio/mp3")

    # Ask user to either upload audio for ASR or type their understanding
    if use_asr:
        st.markdown(f"**Record your verbal understanding of Paragraph {para_num}:**")
        audio_file = st.file_uploader(
            f"Upload a short audio (WAV/MP3) for Para {para_num}:",
            type=["wav", "mp3"],
            key=f"audio_{para_num}"
        )
        if audio_file:
            audio_bytes = audio_file.read()
            asr_text = transcribe_whisper(audio_bytes)
            st.markdown(f"> **You said:** {asr_text}")
            st.session_state.user_data["para_asr_text"][para_num] = asr_text

            gold_summary = selected_summaries[str(para_num)]
            para_score = score_paraphrase(asr_text, gold_summary)
            if para_score >= 10:
                st.success("Excellent! You captured the main idea.")
            elif para_score >= 5:
                st.warning("Close — you missed some nuances; see gold summary below.")
                st.info(f"**Gold Summary:** {gold_summary}")
            else:
                st.error("You seem to have missed the core idea. Here’s the gold summary:")
                st.info(f"**Gold Summary:** {gold_summary}")
    else:
        st.markdown(f"**Type your understanding of Paragraph {para_num}:**")
        user_typed = st.text_input(
            f"Your summary for Para {para_num}:",
            key=f"type_para_{para_num}"
        )
        if user_typed:
            st.session_state.user_data["paragraph_understandings"][para_num] = user_typed
            gold_summary = selected_summaries[str(para_num)]
            para_score = score_paraphrase(user_typed, gold_summary)
            if para_score >= 10:
                st.success("Excellent! You captured the main idea.")
            elif para_score >= 5:
                st.warning("Close — you missed some nuances; see gold summary below.")
                st.info(f"**Gold Summary:** {gold_summary}")
            else:
                st.error("You seem to have missed the core idea. Here’s the gold summary:")
                st.info(f"**Gold Summary:** {gold_summary}")

# ---------------------------------------------------------
# 12) Step 3: Paragraph Matching for Each Question
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 3: For Each Question, Match to a Paragraph")

for q in selected_questions:
    qid = q["id"]
    st.markdown(f"**{qid}. {q['text']}**")
    choice = st.selectbox(
        f"Which paragraph does {qid} refer to?",
        [f"Paragraph {i}" for i in selected_passage["paragraphs"].keys()],
        key=f"match_{qid}"
    )
    st.session_state.user_data["question_paragraph_match"][qid] = int(choice.split()[-1])

# ---------------------------------------------------------
# 13) Step 4: Inference & Vocabulary Drills
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 4: Inference & Vocabulary Drills")

for q in selected_questions:
    qid = q["id"]
    if q["type"] == "inference":
        st.markdown(f"**{qid}. (Inference) {q['text']}**")
        ua = st.text_input(f"Your inferred answer for {qid}:", key=f"inference_{qid}")
        st.session_state.user_data["inference_answers"][qid] = ua
    elif q["type"] == "vocab":
        para_id = q["gold_paragraph_id"]
        st.markdown(f"**{qid}. (Vocabulary) {q['text']}**")
        st.markdown(f"> **Excerpt from Paragraph {para_id}:** {selected_passage['paragraphs'][str(para_id)]}")
        vp = st.text_input(f"Type your paraphrase/definition of the word for {qid}:", key=f"vocab_{qid}")
        st.session_state.user_data["vocab_paraphrases"][qid] = vp

# ---------------------------------------------------------
# 14) Step 5: Paraphrase What Each Question Is Asking
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 5: Paraphrase What Each Question Is Asking")

for q in selected_questions:
    qid = q["id"]
    st.markdown(f"**{qid}. {q['text']}**")
    qp = st.text_input(f"Paraphrase {qid} in your own words:", key=f"paraphrase_q_{qid}")
    st.session_state.user_data["question_paraphrases"][qid] = qp

# ---------------------------------------------------------
# 15) Step 6: Submit & Score Everything
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Step 6: Submit All Answers & View Your Scores")

if st.button("Submit All Answers"):
    ud = st.session_state.user_data

    # 15.1 Paragraph-Matching Score
    pm_scores = []
    for q in selected_questions:
        qid = q["id"]
        user_choice = ud["question_paragraph_match"].get(qid, 0)
        gold_para = q["gold_paragraph_id"]
        pm_scores.append(10 if user_choice == gold_para else 0)
    avg_pm = np.mean(pm_scores) if pm_scores else 0

    # 15.2 Inference Scores
    inf_scores = []
    inf_questions = [qq for qq in selected_questions if qq["type"] == "inference"]
    if inf_questions:
        gold_infs = [qq["gold_inference_answer"] for qq in inf_questions]
        gold_inf_embs = nlp_model.encode(gold_infs, convert_to_tensor=True)
        for idx, qq in enumerate(inf_questions):
            qid = qq["id"]
            ua = ud["inference_answers"].get(qid, "")
            if ua:
                ua_emb = nlp_model.encode(ua, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(ua_emb, gold_inf_embs[idx:idx+1]).item()
                if sim > 0.75:
                    inf_scores.append(10)
                elif sim > 0.5:
                    inf_scores.append(5)
                else:
                    inf_scores.append(0)
            else:
                inf_scores.append(0)
    avg_inf = np.mean(inf_scores) if inf_scores else 0

    # 15.3 Vocabulary Scores
    vocab_scores = []
    vocab_questions = [qq for qq in selected_questions if qq["type"] == "vocab"]
    if vocab_questions:
        gold_vocs = [qq["gold_vocab_definition"] for qq in vocab_questions]
        gold_voc_embs = nlp_model.encode(gold_vocs, convert_to_tensor=True)
        for idx, qq in enumerate(vocab_questions):
            qid = qq["id"]
            ua = ud["vocab_paraphrases"].get(qid, "")
            if ua:
                ua_emb = nlp_model.encode(ua, convert_to_tensor=True)
                sim = util.pytorch_cos_sim(ua_emb, gold_voc_embs[idx:idx+1]).item()
                if sim > 0.75:
                    vocab_scores.append(10)
                elif sim > 0.5:
                    vocab_scores.append(5)
                else:
                    vocab_scores.append(0)
            else:
                vocab_scores.append(0)
    avg_vocab = np.mean(vocab_scores) if vocab_scores else 0

    # 15.4 Question-Paraphrase Scores
    qp_scores = []
    for qq in selected_questions:
        qid = qq["id"]
        up = ud["question_paraphrases"].get(qid, "")
        if up:
            up_emb = nlp_model.encode(up, convert_to_tensor=True)
            gq_emb = nlp_model.encode(qq["text"], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(up_emb, gq_emb).item()
            if sim > 0.7:
                qp_scores.append(10)
            elif sim > 0.4:
                qp_scores.append(5)
            else:
                qp_scores.append(0)
        else:
            qp_scores.append(0)
    avg_qp = np.mean(qp_scores) if qp_scores else 0

    # 15.5 Paragraph-Understanding Scores
    pu_scores = []
    for idx, gs in selected_summaries.items():
        para_num = int(idx)
        # Prioritize ASR text if present
        user_para_text = ud["para_asr_text"].get(para_num, ud["paragraph_understandings"].get(para_num, ""))
        if user_para_text:
            pu_scores.append(score_paraphrase(user_para_text, gs))
        else:
            pu_scores.append(0)
    avg_pu = np.mean(pu_scores) if pu_scores else 0

    # 15.6 Keyword-Flagging Scores
    kw_scores = []
    for qq in selected_questions:
        qid = qq["id"]
        flagged = ud["keyword_flags"].get(qid, [])
        q_words = set(qq["text"].lower().split())
        if any(kw for kw in flagged if kw in q_words):
            kw_scores.append(10)
        else:
            kw_scores.append(0)
    avg_kw = np.mean(kw_scores) if kw_scores else 0

    # Display Summary Table
    st.markdown("## 📝 Your Overall Scores")
    summary_table = {
        "Metric": [
            "Paragraph Matching",
            "Inference",
            "Vocabulary",
            "Question Paraphrase",
            "Paragraph Understanding",
            "Keyword Flagging"
        ],
        "Score (0–10)": [
            round(avg_pm, 1),
            round(avg_inf, 1),
            round(avg_vocab, 1),
            round(avg_qp, 1),
            round(avg_pu, 1),
            round(avg_kw, 1)
        ]
    }
    st.table(summary_table)
    st.success("✅ Scoring completed! Check above for details.")

st.markdown("---")
st.info("MVP by YourName • Powered by Streamlit + Sentence-Transformers + OpenAI Whisper + Google TTS")
