#!/usr/bin/env python
# evaluate_summarizer.py - Evaluate summarization model with ROUGE and BERT scores

import pandas as pd
import torch
import argparse
import os
import re
import numpy as np
from tqdm import tqdm
import nltk
import evaluate
import glob
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AutoTokenizer,
    AutoModel
)
from datasets import Dataset
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --- Reproducible split config ---
TEST_SIZE = 0.2
SEED = 42

# Initialize ROUGE metrics
nltk.download('punkt', quiet=True)
rouge_metric = evaluate.load("rouge")

# BERT model for semantic similarity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_tags(text):
    """Remove all tags from text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/[A-Z]+>', '', text)
    text = re.sub(r'[A-Z]+>', '', text)
    text = re.sub(r'/[A-Z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_model_and_tokenizer(model_path, model_type="bart"):
    """Load the model and tokenizer from the given path."""
    if model_type.lower() == "bart":
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizerFast.from_pretrained(model_path)
    elif model_type.lower() == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, tokenizer

def load_bert_model():
    """Load pre-trained BERT model for semantic similarity."""
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_bert_similarity(texts1, texts2, bert_model, bert_tokenizer):
    """Compute semantic similarity between pairs of texts using BERT."""
    if hasattr(texts1, 'tolist'):
        texts1 = texts1.tolist()
    if hasattr(texts2, 'tolist'):
        texts2 = texts2.tolist()
    if not texts1 or not texts2:
        return []
    encoded_input = bert_tokenizer(
        texts1 + texts2,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    emb1 = sentence_embeddings[:len(texts1)]
    emb2 = sentence_embeddings[len(texts1):]
    sims = []
    for i in range(len(texts1)):
        sim = cosine_similarity(
            emb1[i].cpu().numpy().reshape(1, -1),
            emb2[i].cpu().numpy().reshape(1, -1)
        )[0][0]
        sims.append(sim)
    return sims

def load_tagged_files(tagged_dir, test_filenames):
    """Load tagged files from directory that match test filenames."""
    tagged_paths = glob.glob(os.path.join(tagged_dir, "*_tagged.txt"))
    rows = []
    test_set = set(test_filenames)
    for path in tagged_paths:
        base = os.path.basename(path).replace("_tagged.txt", ".json")
        if base in test_set:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            rows.append({'FILENAME': base, 'input_text': content})
    return pd.DataFrame(rows)

def generate_summaries(model, tokenizer, inputs, max_input_length=1024,
                       max_output_length=256, batch_size=4, beam_size=4):
    model.to(device)
    model.eval()
    gen = []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating"):
            batch = inputs['input_text'][i:i+batch_size].tolist()
            tok = tokenizer(batch,
                            max_length=max_input_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt').to(device)
            out = model.generate(
                input_ids=tok['input_ids'],
                attention_mask=tok['attention_mask'],
                max_length=max_output_length,
                num_beams=beam_size,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            gen.extend([clean_tags(d) for d in dec])
    return gen

def compute_rouge_scores(preds, refs):
    """Compute corpus and per-example ROUGE."""
    if hasattr(preds, 'tolist'):
        preds = preds.tolist()
    if hasattr(refs, 'tolist'):
        refs = refs.tolist()
    if not preds or not refs:
        return {}, []
    fmt_p = ["\n".join(nltk.sent_tokenize(p.strip())) for p in preds]
    fmt_r = ["\n".join(nltk.sent_tokenize(r.strip())) for r in refs]
    corpus = rouge_metric.compute(predictions=fmt_p, references=fmt_r, use_stemmer=True)
    corpus = {k: v*100 for k,v in corpus.items()}
    per_ex = []
    for p, r in zip(fmt_p, fmt_r):
        ex = rouge_metric.compute(predictions=[p], references=[r], use_stemmer=True)
        per_ex.append({k: v*100 for k,v in ex.items()})
    return corpus, per_ex

def display_comparison(fnames, ideals, gens, num=5):
    data = []
    for i in range(min(num, len(gens))):
        data.append([fnames[i], ideals[i], gens[i]])
    print(tabulate(data, headers=["File","Reference","Generated"], tablefmt="grid"))

def save_results(df, gens, metrics, out_path):
    res = df.copy().reset_index(drop=True)
    res['generated'] = gens
    if 'per_example_rouge' in metrics:
        for idx, scores in enumerate(metrics['per_example_rouge']):
            for m,v in scores.items():
                col = m if m in res.columns else m
                res.at[idx, col] = v
    if 'bert_similarity' in metrics:
        res['bert_similarity'] = metrics['bert_similarity']
    res.to_csv(out_path, index=False)
    print(f"Saved {len(res)} rows to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate summarizer")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", choices=["bart","t5"], default="bart")
    parser.add_argument("--csv_path", default="data.csv")
    parser.add_argument("--tagged_dir", default="./tagged")
    parser.add_argument("--output_path", default="evaluation_results.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    # --- Step 1: Load and split CSV ---
    df = pd.read_csv(args.csv_path, dtype=str)
    required = ['FILENAME','IDEAL INPUT','OUTPUT']
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV needs {required}")
    df = df.dropna(subset=['IDEAL INPUT','OUTPUT']).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    test_df = test_df.reset_index(drop=True)
    test_df = test_df.rename(columns={
        'IDEAL INPUT':'input_text',
        'OUTPUT':'target_text'
    })[['FILENAME','input_text','target_text']]

    print(f"CSV split → {len(train_df)} train / {len(test_df)} test")

    # --- Step 2: Load any tagged files matching the test split ---
    test_files = test_df['FILENAME'].tolist()
    tagged_df = load_tagged_files(args.tagged_dir, test_files)
    if tagged_df.empty:
        print("No tagged files found for this test split.")
    else:
        # attach the reference summaries
        ref_map = dict(zip(test_df['FILENAME'], test_df['target_text']))
        tagged_df['target_text'] = tagged_df['FILENAME'].map(ref_map).fillna("")
        tagged_df = tagged_df.reset_index(drop=True)

    # --- Step 3: Load models ---
    print("Loading models…")
    bert_model, bert_tokenizer = load_bert_model()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_type)

    # --- Step 4: Generate summaries ---
    # 4a) For raw CSV test inputs
    print("Generating for CSV test set…")
    csv_gens = generate_summaries(
        model, tokenizer,
        test_df,
        batch_size=args.batch_size,
        beam_size=args.beam_size
    )

    # 4b) For tagged inputs (if any)
    if not tagged_df.empty:
        print("Generating for tagged files…")
        tagged_gens = generate_summaries(
            model, tokenizer,
            tagged_df,
            batch_size=args.batch_size,
            beam_size=args.beam_size
        )
    else:
        tagged_gens = []

    # --- Step 5: Compute metrics ---
    print("Computing ROUGE & BERT similarity…")
    csv_rouge, csv_per = compute_rouge_scores(csv_gens, test_df['target_text'])
    csv_bert = compute_bert_similarity(csv_gens, test_df['target_text'], bert_model, bert_tokenizer)

    if not tagged_df.empty:
        tagged_rouge, tagged_per = compute_rouge_scores(tagged_gens, tagged_df['target_text'])
        tagged_bert = compute_bert_similarity(tagged_gens, tagged_df['target_text'], bert_model, bert_tokenizer)

    # --- Step 6: Display & save ---
    print("\n=== CSV Test Set Results ===")
    for m,v in csv_rouge.items():
        print(f"  {m}: {v:.4f}")
    print(f"  BERT sim: {np.mean(csv_bert):.4f}")
    display_comparison(test_df['FILENAME'], test_df['target_text'], csv_gens, num=args.num_examples)
    save_results(test_df, csv_gens, {'per_example_rouge': csv_per, 'bert_similarity': csv_bert},
                 args.output_path)

    if not tagged_df.empty:
        print("\n=== Tagged Files Results ===")
        for m,v in tagged_rouge.items():
            print(f"  {m}: {v:.4f}")
        print(f"  BERT sim: {np.mean(tagged_bert):.4f}")
        display_comparison(tagged_df['FILENAME'], tagged_df['target_text'], tagged_gens, num=args.num_examples)
        tagged_out = args.output_path.replace('.csv', '_tagged.csv')
        save_results(tagged_df, tagged_gens,
                     {'per_example_rouge': tagged_per, 'bert_similarity': tagged_bert},
                     tagged_out)

if __name__ == "__main__":
    main()
