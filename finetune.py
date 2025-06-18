import os
import logging
import pandas as pd
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fine-tuning")

# Load evaluation metrics
rouge = evaluate.load("rouge")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = None

def prepare_dataset_from_csv(csv_path):
    logger.info(f"Loading dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Initial dataframe shape: {df.shape}")
    df = df.dropna(subset=["Question", "Answer"])
    df = df[df["Question"].str.strip() != ""]
    df = df[df["Answer"].str.strip() != ""]
    logger.info(f"Filtered to {len(df)} valid rows.")
    df["input_text"] = "question: " + df["Question"]
    df["target_text"] = df["Answer"]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Training size: {len(train_df)}, Validation size: {len(val_df)}")
    return Dataset.from_pandas(train_df.reset_index(drop=True)), Dataset.from_pandas(val_df.reset_index(drop=True))

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Handle tuple output
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    # Flatten if needed
    def flatten_preds(preds):
        if isinstance(preds[0], list) and isinstance(preds[0][0], list):
            return [p[0] if isinstance(p[0], list) else p for p in preds]
        return preds

    preds = flatten_preds(preds)
    labels = flatten_preds(labels)

    # --- Correction : filtrer les IDs invalides ---
    def filter_valid_ids(seq):
        return [int(i) for i in seq if isinstance(i, int) and 0 <= int(i) < tokenizer.vocab_size]

    preds = [filter_valid_ids(p) for p in preds]
    labels_for_decode = [
        [l if l != -100 and 0 <= l < tokenizer.vocab_size else tokenizer.pad_token_id for l in label]
        for label in labels
    ]

    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

    # Exact match accuracy
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
    accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0.0

    # ToC:\Users\gaied\Desktop\pfe\chatbot\FSTT_FT_DATA.csvken-level F1
    preds_flat = [token for pred in preds for token in pred]
    labels_flat = [token for label in labels for token in label]
    filtered_preds = [p for p, l in zip(preds_flat, labels_flat) if l != -100]
    filtered_labels = [l for l in labels_flat if l != -100]
    f1 = f1_metric.compute(predictions=filtered_preds, references=filtered_labels, average="macro")["f1"]

    # ROUGE-L
    rougeL = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]

    logger.info(f"Sample prediction: {decoded_preds[0]}")
    logger.info(f"Sample reference:  {decoded_labels[0]}")
    logger.info(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROUGE-L: {rougeL:.4f}")

    return {"accuracy": accuracy, "f1": f1, "rougeL": rougeL}

def finetune_lora_from_csv(csv_path, output_dir="qa_finetune_lora"):
    global tokenizer

    logger.info("Preparing dataset...")
    train_dataset, val_dataset = prepare_dataset_from_csv(csv_path)

    logger.info("Loading model and tokenizer...")
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    logger.info("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)

    def preprocess_function(examples):
        inputs = tokenizer(
            examples["input_text"], max_length=512, truncation=True, padding="max_length"
        )
        targets = tokenizer(
            examples["target_text"], max_length=128, truncation=True, padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    logger.info("Tokenizing training dataset...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    logger.info("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    logger.info("Setting up data collator and training arguments...")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=output_dir,



        
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=8,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating final model...")
    final_metrics = trainer.evaluate()
    logger.info("📊 Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    csv_path = r"C:\Users\Laptek Store\Desktop\chatbot\FSTT_FT_DATA.csv"
    finetune_lora_from_csv(csv_path)