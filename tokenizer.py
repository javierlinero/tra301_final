import json
import os
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from utility import identify_novel_sentences_with_structure, filter_contribution_sentences

def load_papers(data_path, whitelist):
    """Load only whitelisted papers and return mapping from filename to JSON data."""
    papers = {}
    for fn in os.listdir(data_path):
        if not fn.endswith('.json') or fn not in whitelist:
            continue
        path = os.path.join(data_path, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            print(f"[WARN] {fn} not UTF-8, falling back to latin-1")
            with open(path, 'r', encoding='latin-1') as f:
                data = json.loads(f.read())
        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse {fn}: {e}")
            continue
        papers[fn] = data
    return papers


def tag_abstract(abstract, novel_sentences):
    """Wrap each sentence in <BACKGROUND> or <CONTRIBUTION> tags."""
    sentences = sent_tokenize(abstract)
    parts = []
    for s in sentences:
        clean = s.strip()
        if clean in novel_sentences:
            parts.append(f"<CONTRIBUTION>{clean}</CONTRIBUTION>")
        else:
            parts.append(f"<BACKGROUND>{clean}</BACKGROUND>")
    return " ".join(parts)


def create_tagged_output(pdata):
    """Generate a single tagged-output string from paper data or None if no contributions."""
    abstract = pdata['main'].get('abstract')
    if not abstract:
        return None
    # Gather predecessor abstracts
    pred_abstracts = []
    for ref in pdata.get('references', []):
        rid = ref.get('paperId')
        if rid in papers_map_by_pid:
            a = papers_map_by_pid[rid]['main'].get('abstract')
            if a:
                pred_abstracts.append(a)
    for ref in pdata.get('references', []):
        a = ref.get('abstract')
        if a and a not in pred_abstracts:
            pred_abstracts.append(a)
    if not pred_abstracts:
        return None
    
    novel = identify_novel_sentences_with_structure(abstract, pred_abstracts)
    novel = filter_contribution_sentences(novel)
    if not novel:
        return None
    

    # CONSTRUCT THE OUTPUT
    tagged = tag_abstract(abstract, novel)
    output = (
        "<TASK> Summarize this abstract for high-school audience while highlight importance of contributions </TASK> "
        "<ABSTRACT> " + tagged + " </ABSTRACT>"
    )
    return output

def main():
    filenames = [
        "ARXIV_1411_7798.json",
        "ARXIV_2103_12820.json",
        "ARXIV_2004_02020.json",
        "ARXIV_1811_08536.json",
        "ARXIV_1506_00902.json",
        "ARXIV_2110_12691.json",
        "ARXIV_2003_03849.json",
        "ARXIV_1010_1662.json",
        "ARXIV_2403_06647.json",
        "ARXIV_1805_12212.json",
        "ARXIV_astro_ph_0205031.json",
        "ARXIV_2206_06633.json",
        "ARXIV_physics_9806027.json",
        "ARXIV_2101_12333.json",
        "ARXIV_1304_4762.json"
    ]

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data", "raw")
    print(f"Loading papers from {data_dir}…")
    papers = load_papers(data_dir, whitelist=filenames)

    # build a mapping from paperId to data for reference lookups
    global papers_map_by_pid
    papers_map_by_pid = {data['main']['paperId']: data for data in papers.values()}

    for fn, pdata in papers.items():
        tagged = create_tagged_output(pdata)
        if not tagged:
            print(f"Skipping {fn}, no novel contributions detected.")
            continue
        
        base_name = os.path.splitext(fn)[0]
        os.makedirs(os.path.join(base_dir, "tagged"), exist_ok=True)
        out_file = os.path.join(base_dir, "tagged", f"{base_name}_tagged.txt")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(tagged)
        print(f"Tagged abstract saved to {out_file}")

if __name__ == "__main__":
    main()
