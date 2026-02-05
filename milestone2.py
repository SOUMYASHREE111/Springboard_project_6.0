import os
import re
import json
import spacy
from typing import List, Dict

# ======================================================
# 🔴 CHANGE PATH ONLY HERE
# ======================================================
RAW_TEXT_DIR = r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\data copy"

# ======================================================
# Load spaCy
# ======================================================
nlp = spacy.load("en_core_web_sm")

# ======================================================
# 1. Clean & Normalize Text
# ======================================================
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,:-]", "", text)
    return text.strip().lower()

# ======================================================
# 2. Read TXT Files
# ======================================================
def read_txt_files(folder_path: str) -> List[Dict]:
    records = []
    doc_id = 1

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            records.append({
                "id": doc_id,
                "source_file": file,
                "content": clean_text(content)
            })
            doc_id += 1

    return records

# ======================================================
# 3. Entity Extraction (spaCy)
# ======================================================
def extract_entities(text: str) -> List[Dict]:
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities

# ======================================================
# 4. Relationship Extraction (simple rule-based)
# ======================================================
def extract_relationships(entities: List[Dict]) -> List[Dict]:
    relationships = []

    persons = [e["text"] for e in entities if e["label"] == "PERSON"]
    orgs = [e["text"] for e in entities if e["label"] == "ORG"]
    locations = [e["text"] for e in entities if e["label"] == "GPE"]

    for p in persons:
        for o in orgs:
            relationships.append({
                "source": p,
                "relation": "associated_with",
                "target": o
            })

    for p in persons:
        for l in locations:
            relationships.append({
                "source": p,
                "relation": "located_in",
                "target": l
            })

    return relationships

# ======================================================
# 5. Triplet Creation
# ======================================================
def create_triplets(relationships: List[Dict]) -> List[List[str]]:
    return [[r["source"], r["relation"], r["target"]] for r in relationships]

# ======================================================
# 6. Main Pipeline
# ======================================================
def process_dataset(records: List[Dict]):
    output_entities = []
    all_relationships = []
    all_triplets = []

    for record in records:
        entities = extract_entities(record["content"])
        relationships = extract_relationships(entities)
        triplets = create_triplets(relationships)

        output_entities.append({
            "id": record["id"],
            "source_file": record["source_file"],
            "text": record["content"],
            "entities": entities
        })

        all_relationships.extend(relationships)
        all_triplets.extend(triplets)

    return output_entities, all_relationships, all_triplets

# ======================================================
# 7. Save JSON
# ======================================================
def save_json(filename: str, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved {filename}")

# ======================================================
# 8. Run
# ======================================================
if __name__ == "__main__":
    records = read_txt_files(RAW_TEXT_DIR)

    if not records:
        print("❌ No TXT files found.")
    else:
        output, relationships, triplets = process_dataset(records)

        save_json("output.json", output)
        save_json("relationship.json", relationships)
        save_json("triplets.json", triplets)

        print("🎉 Milestone-2 completed")
