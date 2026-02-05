import os
import re
import pandas as pd
from typing import List, Dict

# =============================
# CONFIGURATION (EDIT ONLY HERE)
# =============================

# Folder where your .txt files are present
RAW_DATA_DIR =r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\data copy"

# Folder where structured.csv will be saved (auto-created)
OUTPUT_DIR =r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\processed_data"

# Output CSV file name
OUTPUT_FILE = "structured.csv"


# =============================
# TEXT NORMALIZATION FUNCTION
# =============================

def normalize_text(text: str) -> str:
    """
    Clean and normalize raw text while preserving line structure
    """
    text = text.lower()

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove extra spaces in each line
    lines = []
    for line in text.split("\n"):
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)

    # Join lines back with newline (vertical format preserved)
    return "\n".join(lines)


# =============================
# DATA INGESTION FUNCTION
# =============================

def ingest_txt_files(folder_path: str) -> List[Dict]:
    records = []
    doc_id = 1

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            cleaned_text = normalize_text(raw_text)

            records.append({
                "doc_id": doc_id,
                "file_name": file_name,
                "cleaned_text": cleaned_text,
                "word_count": len(cleaned_text.split())
            })

            doc_id += 1

    return records


# =============================
# MAIN PIPELINE
# =============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Reading files from:", RAW_DATA_DIR)
    data = ingest_txt_files(RAW_DATA_DIR)
    print("Number of documents read:", len(data))

    if not data:
        print("❌ No .txt files found in raw_data folder")
        return

    df = pd.DataFrame(data)
    print(df.head())

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False, sep=',', quoting=csv.QUOTE_ALL)
    print(f"✅ structured.csv saved at: {output_path}")



# =============================
# ENTRY POINT
# =============================

if __name__ == "__main__":
    main()
