import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # Used if not using predefined lists
from transformers import (
    BartForConditionalGeneration,  # Changed from T5ForConditionalGeneration
    BartTokenizerFast,             # Changed from T5TokenizerFast
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, DatasetDict
import evaluate
import nltk
import os
import numpy as np
import re

# --- Configuration ---
CSV_PATH = "data.csv"  # Path to your CSV file
MODEL_NAME = "facebook/bart-large-cnn"  # Correct HF model name for BART Large CNN
OUTPUT_DIR = "./summarizer_model"  # Where to save the trained model
LOGGING_DIR = "./logs"  # Directory for TensorBoard logs

# Define train/test split based on filenames
TRAIN_FILENAMES = [
    "ARXIV_1411_7798.json", "ARXIV_2103_12820.json", "ARXIV_2004_02020.json",
    "ARXIV_1811_08536.json", "ARXIV_1506_00902.json", "ARXIV_2110_12691.json",
    "ARXIV_2003_03849.json", "ARXIV_1010_1662.json", "ARXIV_2403_06647.json",
    "ARXIV_1805_12212.json"
]
TEST_FILENAMES = [
    "ARXIV_astro_ph_0205031.json", "ARXIV_2206_06633.json",
    "ARXIV_physics_9806027.json", "ARXIV_2101_12333.json",
    "ARXIV_1304_4762.json"
]

# Training parameters
MAX_INPUT_LENGTH = 1024  # Max length for input tokens
MAX_TARGET_LENGTH = 256   # Max length for output summary tokens
BATCH_SIZE = 4           # Adjust based on GPU memory
NUM_EPOCHS = 20          # Number of training epochs (adjust as needed)
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STEPS = 100         # Evaluate every N steps
SAVE_STEPS = 200         # Save checkpoint every N steps

# For ROUGE computation
nltk.download('punkt', quiet=True)
rouge_metric = evaluate.load("rouge")

# --- Helper Functions ---

def clean_tags(text):
    """Remove all tags from text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/[A-Z]+>', '', text)
    text = re.sub(r'[A-Z]+>', '', text)
    text = re.sub(r'/[A-Z]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Extract only the summary block, fallback to stripping tags
_extract_summary_re = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL)
def extract_summary(full: str) -> str:
    """Pull out text inside <SUMMARY>…</SUMMARY>, or strip tags otherwise."""
    m = _extract_summary_re.search(full)
    if m:
        return m.group(1).strip()
    return clean_tags(full)

# --- Data Loading and Preprocessing ---
def load_and_split_data(csv_path, train_files, test_files):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        exit(1)

    if not all(col in df.columns for col in ['FILENAME', 'IDEAL INPUT', 'OUTPUT']):
        print("Error: CSV must contain columns 'FILENAME','IDEAL INPUT','OUTPUT'")
        exit(1)

    df = df.astype({'FILENAME': str, 'IDEAL INPUT': str, 'OUTPUT': str})
    df.dropna(subset=['IDEAL INPUT', 'OUTPUT'], inplace=True)

    train_df = df[df['FILENAME'].isin(train_files)].copy()
    test_df = df[df['FILENAME'].isin(test_files)].copy()

    train_df.rename(columns={'IDEAL INPUT': 'input_text', 'OUTPUT': 'target_text'}, inplace=True)
    test_df.rename(columns={'IDEAL INPUT': 'input_text', 'OUTPUT': 'target_text'}, inplace=True)

    return DatasetDict({
        'train': Dataset.from_pandas(train_df[['input_text', 'target_text']]),
        'test': Dataset.from_pandas(test_df[['input_text', 'target_text']])
    })


def preprocess_function(examples, tokenizer):
    # Input keeps your <TASK> and <ABSTRACT> tags
    inputs = examples['input_text']
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # For BART, we don't need to wrap in a special tag
    # Directly clean target texts
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
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean any remaining tags
    decoded_preds = [clean_tags(x) for x in decoded_preds]
    decoded_labels = [clean_tags(x) for x in decoded_labels]

    decoded_preds = ["\n".join(nltk.sent_tokenize(x)) for x in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(x)) for x in decoded_labels]

    result = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    result = {k: v * 100 for k, v in result.items()}
    lengths = [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
    result['gen_len'] = np.mean(lengths)
    return {k: round(v, 4) for k, v in result.items()}

# --- Main Training ---
if __name__ == "__main__":
    raw_datasets = load_and_split_data(CSV_PATH, TRAIN_FILENAMES, TEST_FILENAMES)

    # Initialize BART tokenizer and model
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = raw_datasets.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=True,
        remove_columns=raw_datasets['train'].column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,  # Use -100 for BART label padding
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
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate(eval_dataset=tokenized['test'])
    print("Evaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    # Generate and print a few summaries
    print("\n--- Generating Summaries for Test Examples ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    samples = tokenized['test'].select(range(min(3, len(tokenized['test']))))

    for i, sample in enumerate(samples):
        input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
        attn = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0
        )
        raw_summary = tokenizer.decode(out[0], skip_special_tokens=True)
        # Clean any remaining tags
        clean_summary = clean_tags(raw_summary)
        
        original_input = raw_datasets['test'][i]['input_text']
        ideal_summary = raw_datasets['test'][i]['target_text']
        
        print(f"\nExample {i+1}:")
        print(f"--- Ideal Summary ---:\n{ideal_summary}")
        print(f"--- Generated Summary ---:\n{clean_summary}")