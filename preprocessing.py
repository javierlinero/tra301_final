import json
import os
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utility import identify_novel_sentences_with_structure, filter_contribution_sentences, enhance_coherence

def load_papers(data_path):
    """Load papers from JSON files under data_path, forcing UTF-8 decoding."""
    papers = {}
    for fn in os.listdir(data_path):
        if not fn.endswith('.json'):
            continue

        path = os.path.join(data_path, fn)
        try:
            # force UTF-8
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        except UnicodeDecodeError:
            # fallback in case it's actually non-UTF8
            print(f"[WARN] {fn} not UTF-8, falling back to latin-1")
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()
                data = json.loads(text)

        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse {fn}: {e}")
            continue

        pid = data['main']['paperId']
        papers[pid] = data

    return papers

def get_predecessor_abstracts(pdata, papers):
    """
    For a single paper's pdata:
    - first grab abstracts from any locally-loaded papers it cites,
    - then grab abstracts straight from the JSON references list if present.
    """
    abstracts = []

    # 1) from your local papers dict
    for ref in pdata.get('references', []):
        rid = ref.get('paperId')
        if rid in papers:
            # use the full JSON-loaded version
            a = papers[rid]['main'].get('abstract')
            if a:
                abstracts.append(a)

    # 2) fallback to the JSON-internal abstract if local file missing
    for ref in pdata.get('references', []):
        a = ref.get('abstract')
        if a and a not in abstracts:
            abstracts.append(a)

    return abstracts

def create_training_examples(papers):
    examples = []
    for pid, pdata in papers.items():
        abstract = pdata['main'].get('abstract')
        if not abstract:
            continue

        # new: grab *all* abstracts, local or embedded
        pred_abstracts = get_predecessor_abstracts(pdata, papers)
        if not pred_abstracts:
            continue

        novel = identify_novel_sentences_with_structure(abstract, pred_abstracts)
        novel = filter_contribution_sentences(novel)

        if not novel:
            # optionally still include an example by skipping this check
            continue

        contrast = enhance_coherence(novel, abstract)
        inp = f"<CONTRAST> {abstract} [SEP] " + " ".join(pred_abstracts)

        examples.append({
            'paper_id': pid,
            'input_text': inp,
            'target_text': contrast
        })

    return examples

def main():
    base_dir  = os.path.dirname(__file__)
    data_dir  = os.path.join(base_dir, "data", "raw")
    out_file  = os.path.join(base_dir, "examples.json")

    print(f"Loading papers from {data_dir}…")
    papers = load_papers(data_dir)
    print(f"  → loaded {len(papers)} papers")

    print("Generating contrastive examples…")
    examples = create_training_examples(papers)
    print(f"  → created {len(examples)} examples")

    print(f"Writing examples to {out_file}…")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

if __name__ == "__main__":
    main()
