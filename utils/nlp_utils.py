# utils/nlp_utils.py

from sentence_transformers import SentenceTransformer, util

# Load the model once (cached by Streamlit, but also safe to load here)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(text_list):
    """
    Given a list of texts, return their embeddings as a tensor.
    """
    return _model.encode(text_list, convert_to_tensor=True)

def score_paraphrase(user_text: str, gold_text: str) -> int:
    """
    Compare user_text against gold_text using cosine similarity.
    Returns an integer score in {0, 5, 10} based on thresholds.
    """
    if not user_text or not gold_text:
        return 0
    u_emb = _model.encode(user_text, convert_to_tensor=True)
    g_emb = _model.encode(gold_text, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(u_emb, g_emb).item()
    if sim > 0.75:
        return 10
    elif sim > 0.5:
        return 5
    else:
        return 0

def best_matching_paragraph(question_keywords: str, paragraph_texts: list) -> (int, float):
    """
    Given a string of question_keywords (comma-separated or full string),
    and a list of paragraph texts, returns:
      (best_paragraph_index (1-based), similarity_score)
    based on cosine similarity between the combined question_keywords and each paragraph.
    """
    if not question_keywords:
        return 0, 0.0
    # Create an embedding for question_keywords
    q_emb = _model.encode(question_keywords, convert_to_tensor=True)
    p_embs = _model.encode(paragraph_texts, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_emb, p_embs).cpu().numpy().flatten()
    if sims.size == 0:
        return 0, 0.0
    best_idx = int(sims.argmax())
    return best_idx + 1, float(sims[best_idx])  # 1-based index

def embed_single(text: str):
    """
    Return the embedding of a single text string.
    """
    return _model.encode(text, convert_to_tensor=True)
