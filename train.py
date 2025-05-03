import os
import re
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, DatasetDict
import evaluate
import nltk

# --- Configuration ---
CSV_PATH = "data.csv"
MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./summarizer_model"
LOGGING_DIR = "./logs"

TEST_SIZE = 0.2
SEED = 42

MAX_INPUT_LENGTH = 1024
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.05
MAX_TARGET_LENGTH = 128
LOGGING_STEPS = 50
EVAL_STEPS = 100
SAVE_STEPS = 200

# For ROUGE computation
nltk.download('punkt', quiet=True)
rouge_metric = evaluate.load("rouge")


# --- Helper Functions ---
def clean_tags(text: str) -> str:
    """Remove all XML-style tags from text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/[A-Z]+>', '', text)
    text = re.sub(r'[A-Z]+>', '', text)
    text = re.sub(r'/[A-Z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

_extract_summary_re = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL)
def extract_summary(full: str) -> str:
    """Pull out text inside <SUMMARY>…</SUMMARY>, or strip tags otherwise."""
    m = _extract_summary_re.search(full)
    if m:
        return m.group(1).strip()
    return clean_tags(full)


def load_and_split_data(csv_path: str, test_size: float, seed: int) -> DatasetDict:
    df = pd.read_csv(csv_path, dtype=str)
    required = ['FILENAME', 'IDEAL INPUT', 'OUTPUT']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df.dropna(subset=['IDEAL INPUT', 'OUTPUT'])
    # reproducible shuffle & split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    # rename cols for our pipeline
    for split_df in (train_df, test_df):
        split_df.rename(
            columns={'IDEAL INPUT': 'input_text', 'OUTPUT': 'target_text'},
            inplace=True
        )

    return DatasetDict({
        'train': Dataset.from_pandas(train_df[['input_text', 'target_text']].reset_index(drop=True)),
        'test':  Dataset.from_pandas(test_df[['input_text', 'target_text']].reset_index(drop=True)),
    })


def preprocess_function(examples, tokenizer):
    inputs = examples['input_text']
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    cleaned_targets = [clean_tags(t) for t in examples['target_text']]
    labels = tokenizer(
        text_target=cleaned_targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding='max_length'
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_pred):
    # unpack
    predictions, labels = eval_pred

    # if somehow we got a tuple (e.g. return_dict_in_generate=True), take the first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # replace any label‐padding in the *labels* with the pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # **NEW**: clamp any negative IDs in *predictions* to pad_token_id
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)

    # now it’s safe to decode
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,      skip_special_tokens=True)

    # clean up any stray tags
    decoded_preds  = [clean_tags(x) for x in decoded_preds]
    decoded_labels = [clean_tags(x) for x in decoded_labels]

    # line‐break separate into sentences for ROUGE
    decoded_preds  = ["\n".join(nltk.sent_tokenize(x)) for x in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(x)) for x in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    result = {k: v for k, v in result.items()}

    # you can still compute average length from your sanitized predictions
    lengths = [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
    result["gen_len"] = np.mean(lengths)

    return {k: round(v, 4) for k, v in result.items()}


# --- Main Training Pipeline ---
if __name__ == "__main__":
    # set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # load & split
    raw_datasets = load_and_split_data(CSV_PATH, test_size=TEST_SIZE, seed=SEED)

    # tokenizer & model
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    # tokenize
    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=raw_datasets['train'].column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_dir=LOGGING_DIR,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Training finished. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate(eval_dataset=tokenized['test'])
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    # Generate a few summaries
    print("\n--- Sample Summaries ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples = tokenized['test'].select(range(min(3, len(tokenized['test']))))

    for i, sample in enumerate(samples):
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
        attn = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            min_length=50,
            max_new_tokens=150,
            max_length=MAX_TARGET_LENGTH,
            num_beams=6,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0
        )
        raw_summary = tokenizer.decode(out[0], skip_special_tokens=True)
        clean_summary = clean_tags(raw_summary)

        print(f"\nExample {i+1}")
        print(f"--- Generated Summary ---\n{clean_summary}")
