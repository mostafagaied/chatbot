# phi2.py (Version finale pour bibliothèques MODERNES)
import os
import logging
import pandas as pd
import torch
import numpy as np
import evaluate

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# --- Configuration ---
# --- Configuration ---
# Use the local path to the model you downloaded
MODEL_NAME = "./phi-2" 
OUTPUT_DIR = "fstt-chatbot-phi-2-lora"
CSV_PATH = r"C:\Users\Laptek Store\Desktop\chatbot\FSTT_FT_DATA.csv" 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("phi-2-finetuning")

rouge = evaluate.load("rouge")
tokenizer = None

def create_chat_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["Question", "Answer"], inplace=True)
    df = df[df["Question"].str.strip() != ""]
    def format_row(row):
        return f"Instruct: {row['Question'].strip()}\nOutput: {row['Answer'].strip()}"
    df["text"] = df.apply(format_row, axis=1)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    return Dataset.from_pandas(train_df[["text"]]), Dataset.from_pandas(val_df[["text"]])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    labels_for_decode = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    cleaned_preds = [pred.split("Output:")[-1].strip() for pred in decoded_preds]
    cleaned_labels = [label.split("Output:")[-1].strip() for label in decoded_labels]
    rouge_results = rouge.compute(predictions=cleaned_preds, references=cleaned_labels)
    exact_match_count = sum(1 for p, l in zip(cleaned_preds, cleaned_labels) if p == l)
    return {"rougeL": rouge_results["rougeL"], "exact_match": exact_match_count / len(cleaned_preds)}

def main():
    global tokenizer
    train_dataset, val_dataset = create_chat_dataset(CSV_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        trust_remote_code=True, device_map="auto"
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["Wqkv", "fc1", "fc2"],
    )

    # Utilisation de la syntaxe MODERNE pour les arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        strategy="epoch",  # Syntaxe moderne
        logging_steps=10,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset,
        eval_dataset=val_dataset, peft_config=peft_config,
        dataset_text_field="text", max_seq_length=256,
        tokenizer=tokenizer, args=training_args,
        packing=False, compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete. Saving best model...")
    final_model_dir = os.path.join(OUTPUT_DIR, "final_best_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

if __name__ == "__main__":
    main()