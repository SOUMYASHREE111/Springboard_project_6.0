import os
import re
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
RAW_DATA_DIR = r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\data copy"
OUTPUT_FILE = r"C:\Users\Soumya Shree\OneDrive\Attachments\milestone_2\cypher.csv"

# Node label for Neo4j
NODE_LABEL = "Player"

# ==============================
# TEXT NORMALIZATION
# ==============================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==============================
# PARSE FILE TO DICTIONARY
# ==============================
def parse_txt_to_dict(file_path: str):
    """
    Converts a text file into a dictionary list.
    Each line is treated as key:value or key,value
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = normalize_text(line)
            if not line:
                continue
            if ':' in line:
                parts = line.split(':')
            elif ',' in line:
                parts = line.split(',')
            else:
                parts = [line]

            name = parts[0].strip()
            attr = parts[1].strip() if len(parts) > 1 else ""
            data_list.append({"name": name, "attribute": attr})
    return data_list


# GENERATE CYPHER QUERIES
# ==============================
def generate_cypher_queries(data_list, node_label=NODE_LABEL):
    queries = []
    for item in data_list:
        # Replace single quotes inside text with escaped single quotes
        name = item["name"].replace("'", "\\'")
        attr = item["attribute"].replace("'", "\\'")
        query = f"CREATE (:{node_label} {{name: '{name}', attribute: '{attr}'}});"
        queries.append({"cypher_query": query})
    return queries


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    all_queries = []

    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".txt") or file_name.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            print(f"Processing: {file_path}")

            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    name = str(row[0]).replace('"', '\\"')
                    attr = str(row[1]).replace('"', '\\"') if len(row) > 1 else ""
                    query = f"CREATE (:{NODE_LABEL} {{name: '{name}', attribute: '{attr}'}});"
                    all_queries.append({"cypher_query": query})

            else:
                data_list = parse_txt_to_dict(file_path)
                all_queries.extend(generate_cypher_queries(data_list))

    # Save to CSV
    if all_queries:
        df_queries = pd.DataFrame(all_queries)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for q in all_queries:
                f.write(q['cypher_query'] + '\n')
 # quoting=1 means quote all
        print(f"✅ Saved {len(df_queries)} Cypher queries to {OUTPUT_FILE}")
    else:
        print("❌ No data found to generate Cypher queries.")

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
