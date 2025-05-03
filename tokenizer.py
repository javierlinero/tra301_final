#!/usr/bin/env python
# tag_abstracts.py — same 80/20 split (seed=42) as training/eval scripts

import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

# your utilities
from utility import identify_novel_sentences_with_structure, filter_contribution_sentences

nltk.download('punkt', quiet=True)

# --- Configurable paths & split ---
CSV_PATH    = "data.csv"            # your master CSV
RAW_DIR     = "subset"
TAGGED_DIR  = "tagged"
TEST_SIZE   = 0.2
RANDOM_SEED = 42


def clean_tags(text):
    # same as before, if needed
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/[A-Z]+>', '', text)
    text = re.sub(r'[A-Z]+>', '', text)
    text = re.sub(r'/[A-Z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_papers(data_path, whitelist):
    """Load only whitelisted papers and return mapping from filename to JSON data."""
    papers = {}
    for fn in os.listdir(data_path):
        if not fn.endswith('.json') or fn not in whitelist:
            continue
        full = os.path.join(data_path, fn)
        try:
            with open(full, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"[WARN] {fn} not UTF-8, falling back to latin-1")
            with open(full, 'r', encoding='latin-1') as f:
                data = json.loads(f.read())
        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse {fn}: {e}")
            continue
        papers[fn] = data
    return papers


def tag_abstract(abstract, novel_sentences):
    """Wrap each sentence in <BACKGROUND> or <CONTRIBUTION> tags."""
    parts = []
    for s in sent_tokenize(abstract):
        clean = s.strip()
        tag = "CONTRIBUTION" if clean in novel_sentences else "BACKGROUND"
        parts.append(f"<{tag}>{clean}</{tag}>")
    return " ".join(parts)


def create_tagged_output(pdata, papers_by_pid):
    abstract = pdata['main'].get('abstract')
    if not abstract:
        return None

    # collect predecessor abstracts
    preds = []
    for ref in pdata.get('references', []):
        pid = ref.get('paperId')
        if pid in papers_by_pid:
            a = papers_by_pid[pid]['main'].get('abstract')
            if a:
                preds.append(a)
    # also any inline abstract field
    for ref in pdata.get('references', []):
        a = ref.get('abstract')
        if a and a not in preds:
            preds.append(a)

    if not preds:
        return None

    novel = identify_novel_sentences_with_structure(abstract, preds)
    novel = filter_contribution_sentences(novel)
    if not novel:
        return None

    tagged = tag_abstract(abstract, novel)
    return (
        "<TASK> Summarize this abstract for high-school audience while highlight importance of contributions </TASK> "
        f"<ABSTRACT> {tagged} </ABSTRACT>"
    )


def main():
    # 1) load CSV and split filenames
    import pandas as pd
    df = pd.read_csv(CSV_PATH, dtype=str)
    if 'FILENAME' not in df.columns:
        raise ValueError("data.csv must contain a 'FILENAME' column")
    df = df.dropna(subset=['FILENAME'])
    train_df, test_df = train_test_split(
        df['FILENAME'].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    whitelist = set(test_df)
    print(f"Tagging {len(whitelist)} files (test split, seed={RANDOM_SEED})")

    # 2) load only those JSONs
    papers = load_papers(RAW_DIR, whitelist)
    # build paperId → data map
    papers_by_pid = {
        pdata['main']['paperId']: pdata
        for pdata in papers.values()
        if pdata['main'].get('paperId')
    }

    # 3) tag each
    os.makedirs(TAGGED_DIR, exist_ok=True)
    for fn, pdata in papers.items():
        tagged_output = create_tagged_output(pdata, papers_by_pid)
        if not tagged_output:
            print(f"Skipping {fn}: no novel contributions found.")
            continue
        base = os.path.splitext(fn)[0]
        out_path = os.path.join(TAGGED_DIR, f"{base}_tagged.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(tagged_output)
        print(f"→ {out_path}")

if __name__ == "__main__":
    main()
