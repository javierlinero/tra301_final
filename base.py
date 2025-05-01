#!/usr/bin/env python
# use_pretrained_bart.py - Generate summaries using pretrained BART CNN without additional training

import pandas as pd
import torch
import argparse
import re
import numpy as np
from tqdm import tqdm
import nltk
import evaluate
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    AutoTokenizer,
    AutoModel
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# --- Split configuration ---
TEST_SIZE = 0.2
SEED = 42

# Initialize ROUGE metrics
nltk.download('punkt', quiet=True)
rouge_metric = evaluate.load("rouge")

# Define default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_tags(text):
    """Remove all tags from text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/[A-Z]+>', '', text)
    text = re.sub(r'[A-Z]+>', '', text)
    text = re.sub(r'/[A-Z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

def compute_bert_similarity(texts1, texts2, bert_model, bert_tokenizer):
    if hasattr(texts1, 'tolist'):
        texts1 = texts1.tolist()
    if hasattr(texts2, 'tolist'):
        texts2 = texts2.tolist()
    if not texts1 or not texts2:
        return []
    enc = bert_tokenizer(texts1 + texts2, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        out = bert_model(**enc)
    embeds = mean_pooling(out, enc['attention_mask'])
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    e1, e2 = embeds[:len(texts1)], embeds[len(texts1):]
    sims = []
    for i in range(len(texts1)):
        sims.append(cosine_similarity(
            e1[i].cpu().numpy().reshape(1,-1),
            e2[i].cpu().numpy().reshape(1,-1)
        )[0][0])
    return sims

def generate_summaries(model, tokenizer, df, max_in=1024, max_out=256, batch_size=4, beam_size=4):
    model.to(device).eval()
    gens = []
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Generating summaries"):
            batch = df['input_text'].iloc[i:i+batch_size].tolist()
            batch = [clean_tags(t) for t in batch]
            tok = tokenizer(batch, max_length=max_in, padding='max_length', truncation=True, return_tensors='pt').to(device)
            out = model.generate(
                input_ids=tok['input_ids'],
                attention_mask=tok['attention_mask'],
                max_length=max_out,
                num_beams=beam_size,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            gens.extend([clean_tags(d) for d in dec])
    return gens

def compute_rouge_scores(preds, refs):
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

def display_comparison(fnames, refs, gens, num=5):
    data = []
    for i in range(min(num, len(gens))):
        data.append([fnames[i], refs[i], gens[i]])
    print(tabulate(data, headers=["File","Reference","Generated"], tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using pretrained BART CNN")
    parser.add_argument("--csv_path", type=str, default="data.csv")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--output_path", type=str, default="pretrained_bart_results.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    # --- Load & split CSV ---
    df = pd.read_csv(args.csv_path, dtype=str)
    required = ['FILENAME','IDEAL INPUT','OUTPUT']
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV needs columns: {required}")
    df = df.dropna(subset=['IDEAL INPUT','OUTPUT']).reset_index(drop=True)

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )
    test_df = test_df.reset_index(drop=True).rename(columns={
        'IDEAL INPUT':'input_text',
        'OUTPUT':'target_text'
    })[['FILENAME','input_text','target_text']]

    print(f"CSV split → {len(test_df)} test examples")

    # Clean targets
    test_df['cleaned_target'] = test_df['target_text'].apply(clean_tags)

    # Load models
    print("Loading models…")
    bert_model, bert_tokenizer = load_bert_model()
    model = BartForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = BartTokenizerFast.from_pretrained(args.model_name)

    # Generate summaries on the test split
    print("Generating summaries on test split…")
    generated = generate_summaries(
        model, tokenizer, test_df,
        batch_size=args.batch_size,
        beam_size=args.beam_size
    )

    # Compute metrics
    print("Computing metrics…")
    rouge_scores, per_example_rouge = compute_rouge_scores(generated, test_df['cleaned_target'])
    bert_sims = compute_bert_similarity(generated, test_df['cleaned_target'], bert_model, bert_tokenizer)

    # Display
    print("\n----- Evaluation Results -----")
    for m, v in rouge_scores.items():
        print(f"  {m}: {v:.4f}")
    print(f"  BERT Semantic Similarity: {np.mean(bert_sims):.4f}")

    print("\n----- Example Summaries -----")
    display_comparison(
        test_df['FILENAME'].tolist(),
        test_df['cleaned_target'].tolist(),
        generated,
        num=args.num_examples
    )

    # Save
    out_df = pd.DataFrame({
        'filename': test_df['FILENAME'],
        'input_text': test_df['input_text'],
        'target_text': test_df['target_text'],
        'generated_summary': generated,
        'bert_similarity': bert_sims
    })
    # add per-example ROUGE columns
    for idx, scores in enumerate(per_example_rouge):
        for metric, score in scores.items():
            col = metric
            if col not in out_df.columns:
                out_df[col] = np.nan
            out_df.at[idx, col] = score

    out_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to {args.output_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
