# rag_pipeline.py

import faiss
import numpy as np
import json
import csv
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------------------
# 1. Load Structured Dataset
# -------------------------------
def load_dataset(folder_path):
    documents = []

    if not os.path.isdir(folder_path):
        raise ValueError("DATA_PATH must be a folder containing .txt files")

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if text:
                documents.append(text)

    return documents


# -------------------------------
# 2. Embedding Generation
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts):
    emb = embedder.encode(texts, convert_to_numpy=True)
    return emb.astype('float32')  


# -------------------------------
# 3. Vector Database (FAISS)
# -------------------------------
class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        embeddings = embeddings.astype('float32')  # <-- ensure type
        self.index.add(embeddings)
        self.texts.extend(texts)


    def search(self, query_embedding, k=3):
        _, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]

    # 🔹 SAVE FAISS INDEX
    def save_index(self, path="faiss.index"):
        faiss.write_index(self.index, path)

    # 🔹 SAVE METADATA
    def save_metadata(self, path="metadata.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": i, "text": t} for i, t in enumerate(self.texts)],
                f,
                indent=2,
                ensure_ascii=False
            )

# -------------------------------
# 4. Semantic Search
# -------------------------------
def semantic_search(query, store, k=3):
    query_emb = generate_embeddings([query])
    return store.search(query_emb, k)

# -------------------------------
# 5. RAG-based Q&A
# -------------------------------
generator = pipeline("text-generation", model="gpt2")

def rag_answer(query, store):
    docs = semantic_search(query, store)
    context = " ".join(docs)

    prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""
    output = generator(prompt, max_length=150)
    return output[0]["generated_text"]

# -------------------------------
# 6. Knowledge Graph (Optional)
# -------------------------------
kg = nx.Graph()
kg.add_edge("RAG", "Retrieval")
kg.add_edge("RAG", "Generation")

def kg_enrichment(query):
    return [node for node in kg.nodes if node.lower() in query.lower()]

# -------------------------------
# 7. Evaluation
# -------------------------------
def evaluate(predicted, reference):
    p_emb = generate_embeddings(predicted)
    r_emb = generate_embeddings(reference)
    return cosine_similarity(p_emb, r_emb).mean()

# 8. MAIN

if __name__ == "__main__":

    DATA_PATH = r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\data copy"

    documents = load_dataset(DATA_PATH)
    print(f"Loaded {len(documents)} documents")

    doc_embeddings = generate_embeddings(documents)

    store = VectorStore(dim=doc_embeddings.shape[1])
    store.add(doc_embeddings, documents)
    print("Number of documents:", len(documents))
    print("FAISS index total vectors:", store.index.ntotal)


    # 🔹 SAVE OUTPUT FILES
    store.save_index("faiss.index")
    store.save_metadata("metadata.json")

    print("\nSaved outputs:")
    print(" - faiss.index")
    print(" - metadata.json")

    # evaluation
    eval_query = "Explain retrieval augmented generation"
    reference_answer = "RAG combines document retrieval with language generation"

    generated_answer = rag_answer(eval_query, store)

    score = evaluate(
        [generated_answer],
        [reference_answer]
    )

    print("\nEvaluation score:", round(score, 3))
# Load FAISS index to check
index = faiss.read_index("faiss.index")
print("FAISS index ntotal:", index.ntotal)

query = "example search query"
query_emb = generate_embeddings([query])
results = store.search(query_emb, k=3)
print("Top results:", results)
