# app.py
# app.py
import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# --- 1. Load Models ---
# =========================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    return embedder, generator

embedder, generator = load_models()

# =========================
# --- 2. Load FAISS Index + Metadata ---
# =========================
@st.cache_data
def load_faiss_metadata():
    # <<< EDIT PATH BELOW IF NEEDED >>>
    index = faiss.read_index("faiss.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_faiss_metadata()

# =========================
# --- 3. Functions ---
# =========================
def embed_text(text):
    return embedder.encode([text], convert_to_numpy=True).astype("float32")

def retrieve_docs(query, k=3):
    q_emb = embed_text(query)
    _, indices = index.search(q_emb, k)
    docs = [metadata[i]["text"] for i in indices[0]]
    return docs

def rag_answer(query, docs):
    context = " ".join(docs)
    context = context[:1500]  # limit context

    prompt = f"""
Answer using context.

Context:
{context}

Question:
{query}
"""
    output = generator(prompt, max_length=200)
    return output[0]["generated_text"]

def evaluate(ans, ref):
    a = embed_text(ans)
    r = embed_text(ref)
    return cosine_similarity(a, r)[0][0]

# =========================
# --- 4. Streamlit UI ---
# =========================
st.set_page_config(page_title="🚀 RAG Knowledge Bot", layout="wide", page_icon="🤖")

# =========================
# --- 4a. Custom CSS for Lavender Theme ---
# =========================
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #E6E6FA; /* Light lavender */
    }

    /* Title */
    h1 {
        color: #4B0082; /* Deep purple */
        text-align: center;
    }

    /* Subtitle */
    p {
        color: #6A5ACD; /* Indigo */
        text-align: center;
    }

    /* Buttons */
    .stButton>button {
        background-color: #9370DB; /* Medium lavender */
        color: white;
        font-weight: bold;
    }

    /* Text input */
    .stTextInput>div>input {
        background-color: #F8F8FF;
        color: #4B0082;
        font-weight: 500;
    }

    /* Markdown boxes */
    .source-box {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        color: #4B0082;
    }
    .answer-box {
        background-color: #D8BFD8;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        color: #4B0082;
        font-size: 16px;
    }
    .eval-box {
        background-color: #FFE4E1;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        color: #800080;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# --- 4b. Page Content ---
# =========================
st.markdown("<h1>🚀 RAG Knowledge Bot</h1>", unsafe_allow_html=True)
st.markdown("<p>Ask anything from your dataset!</p>", unsafe_allow_html=True)

query = st.text_input("💬 Enter your question here:")

if st.button("✨ Get Answer") and query:

    with st.spinner("🤔 Thinking..."):
        docs = retrieve_docs(query)
        answer = rag_answer(query, docs)

        # optional evaluation
        reference = "This is expected answer"
        score = evaluate(answer, reference)

    # Display Answer
    st.markdown(
        f"<div class='answer-box'><strong>💡 Answer:</strong><br>{answer}</div>",
        unsafe_allow_html=True
    )

    # Display Evaluation
    st.markdown(
        f"<div class='eval-box'><strong>📊 Evaluation Score:</strong> {round(score,3)} | 📚 Sources Used: {len(docs)}</div>",
        unsafe_allow_html=True
    )

    # Display Sources
    for i, doc in enumerate(docs):
        st.markdown(
            f"<div class='source-box'>📄 <strong>Source {i+1}:</strong> {doc[:500]}...</div>",
            unsafe_allow_html=True
        )
