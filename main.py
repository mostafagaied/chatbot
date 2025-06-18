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

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("finetune")

def prepare_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Question", "Answer"])
    df = df[df["Question"].str.strip() != ""]
    df = df[df["Answer"].str.strip() != ""]
    df["input_text"] = "question: " + df["Question"]
    df["target_text"] = df["Answer"]
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    return Dataset.from_pandas(train_df.reset_index(drop=True)), Dataset.from_pandas(val_df.reset_index(drop=True))

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # --- Correction : flatten si besoin ---
    if len(preds) > 0 and isinstance(preds[0], list) and isinstance(preds[0][0], list):
        preds = [p[0] for p in preds]
    labels_for_decode = [
        [l if l != -100 else tokenizer.pad_token_id for l in label]
        for label in labels
    ]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
    accuracy = exact_matches / len(decoded_preds) if decoded_preds else 0.0
    preds_flat = [token for pred_list in preds for token in pred_list]
    labels_flat = [token for label_list in labels for token in label_list]
    filtered_preds = [p for p, l in zip(preds_flat, labels_flat) if l != -100]
    filtered_labels = [l for l in labels_flat if l != -100]
    f1 = f1_metric.compute(predictions=filtered_preds, references=filtered_labels, average="macro")["f1"]
    rougeL = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
    logger.info(f"Sample prediction: '{decoded_preds[0]}'")
    logger.info(f"Sample reference:  '{decoded_labels[0]}'")
    logger.info(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROUGE-L: {rougeL:.4f}")
    return {"accuracy": accuracy, "f1": f1, "rougeL": rougeL}

def finetune_lora_from_csv(csv_path, output_dir="qa_finetune_lora_t5"):
    global tokenizer
    train_dataset, val_dataset = prepare_dataset_from_csv(csv_path)
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.gradient_checkpointing_enable()
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
            examples["input_text"], max_length=256, truncation=True, padding="max_length"
        )
        targets = tokenizer(
            examples["target_text"], max_length=64, truncation=True, padding="max_length"
        )
        inputs["labels"] = targets["input_ids"]
        return inputs
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-4,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # IMPORTANT pour Windows !
        dataloader_pin_memory=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    final_metrics = trainer.evaluate()
    logger.info("📊 Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")

if __name__ == "__main__":
    rouge_metric = evaluate.load("rouge")
    f1_metric = evaluate.load("f1")
    csv_file_path = r"C:\Users\Laptek Store\Desktop\chatbot\FSTT_FT_DATA.csv"
    if not os.path.exists(csv_file_path):
        logger.error(f"FATAL: The file was not found at the specified path: {csv_file_path}")
    else:
        finetune_lora_from_csv(csv_file_path)