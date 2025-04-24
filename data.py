import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

DIR = os.path.dirname(__file__)
API_KEY = os.getenv("S2_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing S2_API_KEY in env")

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SAMPLE_SIZE = 100000
MAX_REFS = 50 
BATCH_SIZE = 100 
FIELDS_MAIN = "paperId,title,abstract,year,authors.name,referenceCount,citationCount"
FIELDS_REF = "paperId,title,abstract,year,authors.name,referenceCount,citationCount"

os.makedirs(RAW_DIR, exist_ok=True)
HEADERS = {"x-api-key": API_KEY}

def safe_get(url, **kwargs):
    """GET with retry-on-429 and a small pause to respect rate limits."""
    while True:
        r = requests.get(url, headers=HEADERS, **kwargs)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2))
            print(f"[GET 429] waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(1.5)
        return r

def safe_post(url, **kwargs):
    """POST with retry-on-429 and a small pause to respect rate limits."""
    while True:
        r = requests.post(url, headers=HEADERS, **kwargs)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2))
            print(f"[POST 429] waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(1.5)
        return r

def chunk_list(lst, chunk_size):
    """Yield successive `chunk_size` slices from `lst`."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

#### FETCHING LOGIC #####

def fetch_paper_with_refs(arxiv_id: str) -> dict: 
    """
    Given an arxiv id we fetch the paper's metadata up to a max amount for api safety
    Returns a dict:
      {
        "paperId": ...,
        "title": ...,
        "abstract": ...,
        "year": ...,
        "authors": [...],
        "references": [ { ... each ref ... }, ... ]
      } 
    """
    url_main = f"https://api.semanticscholar.org/graph/v1/paper/{arxiv_id}"
    resp = safe_get(url_main, params={"fields": FIELDS_MAIN})
    main = resp.json()

    result = {
        "main": {
            "paperId":       main.get("paperId"),
            "title":         main.get("title"),
            "abstract":      main.get("abstract"),
            "year":          main.get("year"),
            "authors":       [a.get("name") for a in main.get("authors", [])],
            "referenceCount": main.get("referenceCount"),
            "citationCount":  main.get("citationCount"),
        },
        "references": []
    }

    url_refs = f"https://api.semanticscholar.org/graph/v1/paper/{arxiv_id}/references"
    resp = safe_get(url_refs, params={"fields": "citedPaper.paperId,isInfluential", "limit": MAX_REFS})
    refs = resp.json().get("data", [])
    infl_ids = [
        r["citedPaper"]["paperId"]
        for r in refs
        if r.get("isInfluential")
    ]

    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    for chunk in chunk_list(infl_ids, BATCH_SIZE):
        resp = safe_post(
            batch_url,
            params={"fields": FIELDS_REF},
            json={"ids": chunk}
        )
        for p in resp.json():
            # some items may be errors, skip non‐dicts
            if not isinstance(p, dict):
                continue

            if not p.get("abstract"):
                continue

            result["references"].append({
                "paperId":       p.get("paperId"),
                "title":         p.get("title"),
                "abstract":      p.get("abstract"),
                "year":          p.get("year"),
                "authors":       [a.get("name") for a in p.get("authors", [])],
                "referenceCount": p.get("referenceCount"),
                "citationCount":  p.get("citationCount"),
            })

    if len(result["references"]) == 0:
        return None

    return result

# Creates the dataframe of ARXIV datapoints
if __name__ == "__main__":
    df = pd.read_json(os.path.join(DIR, "data", "arxiv-metadata-oai-snapshot.json"), lines=True)
    ids = df["id"].sample(SAMPLE_SIZE, random_state=132).tolist()
    s2_ids = [f"ARXIV:{i}" for i in ids]


    checkpoint_path = os.path.join("data_checkpoint.txt")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            start_idx = int(f.read().strip() or 0)
    else:
        start_idx = 0

    total = len(s2_ids)
    print(f"Resuming from iteration {start_idx} / {total}")

    pbar = tqdm(total=total, desc="Fetching papers")
    pbar.update(start_idx)

    for idx in range(start_idx, total):
        arxiv_id = s2_ids[idx]
        try:
            rec = fetch_paper_with_refs(arxiv_id)
            if rec is not None:
                out_file = os.path.join(RAW_DIR, f"{arxiv_id.replace(':', '_').replace('/', '_').replace('-','_').replace('.', '_')}.json")
                with open(out_file, "w", encoding="utf-8") as fout:
                    json.dump(rec, fout, ensure_ascii=False, indent=2)
            
            with open(checkpoint_path, "w") as f:
                f.write(str(idx + 1))
        except Exception as e:
            print(f"Error on {arxiv_id}: {e}")
            with open(checkpoint_path, "w") as f:
                f.write(str(idx + 1))
        finally:
            pbar.update(1)

    pbar.close()






