# app.py
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict

# --- 1. Load your dataset ---
@st.cache_data
def load_dataset(path=r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\processed_data\structured.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove any spaces
    return df

df = load_dataset()

@dataclass
class Document:
    content: str
    metadata: Dict

# --- 2. Simple keyword search RAG ---
class SimpleRAG:
    def __init__(self, dataframe):
        self.df = dataframe

    def query(self, user_question: str) -> Dict:
        q = user_question.lower().strip()
        matched_rows = self.df[self.df['cleaned_text'].str.lower().str.contains(q, na=False)]

        if not matched_rows.empty:
            row = matched_rows.iloc[0]  # first match
            answer = row['cleaned_text'][:300] + "..."  # show snippet
            source = row['file_name']
            eval_score = "High 🟢"
            sources = [Document(content=answer, metadata={"source": source, "page": 1})]
        else:
            answer = "Sorry, I couldn't find an answer in the dataset."
            sources = []
            eval_score = "Low 🔴"

        return {"result": answer, "source_documents": sources, "score": eval_score}

# --- 3. Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot (Keyword Search)")

# Initialize RAG
if "rag" not in st.session_state:
    st.session_state.rag = SimpleRAG(df)
    st.session_state.chat_history = []

# User input
query = st.text_input("Ask a question:", "")

if query:
    answer_placeholder = st.empty()
    sources_placeholder = st.empty()

    res = st.session_state.rag.query(query)
    answer_text = res['result']
    source_docs = res['source_documents']
    eval_score = res['score']

    # Show direct answer
    answer_placeholder.markdown(f"**💬 Answer:** {answer_text}")

    # Show sources + evaluation
    if source_docs:
        sources_md = f"**📊 Evaluation Score:** {eval_score}\n\n**📚 Sources:**"
        for doc in source_docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "1")
            sources_md += f"\n- `{src}` (Pg {page})"
    else:
        sources_md = f"**📊 Evaluation Score:** {eval_score}\n⚠️ No source documents found."

    sources_placeholder.markdown(sources_md)

    # Save chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer_text,
        "sources": source_docs,
        "score": eval_score
    })

# --- 4. Display chat history ---
st.markdown("### 🕑 Previous Questions")
for chat in st.session_state.chat_history[::-1]:
    st.markdown(f"**📝 Question:** {chat['question']}")
    st.markdown(f"**💬 Answer:** {chat['answer']}")
    if chat["sources"]:
        st.markdown(f"**📊 Evaluation Score:** {chat['score']}")
        st.markdown("**📚 Sources:**")
        for doc in chat["sources"]:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "1")
            st.markdown(f"- `{src}` (Pg {page})")
    else:
        st.markdown(f"**📊 Evaluation Score:** {chat['score']}")
        st.warning("⚠️ No source documents found.")
    st.markdown("---")
