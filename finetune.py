import os
import logging
import pandas as pd
import numpy as np
from functools import partial

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

# --- 1. Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fine-tuning")

# --- 2. Chargement des Métriques ---
rouge_metric = evaluate.load("rouge")

# --- 3. Préparation et NETTOYAGE du Dataset ---
def prepare_dataset_from_csv(csv_path):
    logger.info(f"Loading dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Initial dataframe shape: {df.shape}")

    # --- NOUVEAU : Processus de nettoyage de données ---
    logger.info("Cleaning and preprocessing data...")
    # 1. Supprimer les lignes avec des valeurs manquantes dans 'Question' ou 'Answer'
    df.dropna(subset=['Question', 'Answer'], inplace=True)

    # 2. Convertir en string pour éviter les erreurs de type
    df['Question'] = df['Question'].astype(str)
    df['Answer'] = df['Answer'].astype(str)

    # 3. Supprimer les espaces de début et de fin
    df['Question'] = df['Question'].str.strip()
    df['Answer'] = df['Answer'].str.strip()

    # 4. Mettre en minuscules pour la cohérence
    df['Question'] = df['Question'].str.lower()
    df['Answer'] = df['Answer'].str.lower()
    
    # 5. Supprimer les lignes où la question ou la réponse est vide après le nettoyage
    df = df[df['Question'] != '']
    df = df[df['Answer'] != '']
    # --- Fin du processus de nettoyage ---

    logger.info(f"Filtered to {len(df)} valid and clean rows.")
    
    # Le préfixe "question:" est très efficace pour les modèles Flan-T5
    df["input_text"] = "question: " + df["Question"]
    df["target_text"] = df["Answer"]
    
    # Division en entraînement et validation
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    logger.info(f"Training size: {len(train_df)}, Validation size: {len(val_df)}")
    
    return Dataset.from_pandas(train_df.reset_index(drop=True)), Dataset.from_pandas(val_df.reset_index(drop=True))

# --- 4. Calcul des Métriques d'Évaluation ---
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    predicted_ids = np.argmax(preds, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

    for i in range(min(2, len(decoded_preds))):
        logger.info(f"Sample {i+1} Prediction: '{decoded_preds[i].strip()}'")
        logger.info(f"Sample {i+1} Reference:  '{decoded_labels[i].strip()}'")

    return {"rougeL": rouge_scores["rougeL"]}

# --- 5. Fonction Principale de Fine-Tuning ---
def finetune_lora_from_csv(csv_path, output_dir):
    logger.info("--- Starting Fine-Tuning Process ---")
    
    train_dataset, val_dataset = prepare_dataset_from_csv(csv_path)

    logger.info("Loading model and tokenizer (flan-t5-base)...")
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    logger.info("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def preprocess_function(examples):
        inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(examples["target_text"], max_length=256, truncation=True)
        inputs["labels"] = targets["input_ids"]
        return inputs

    logger.info("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    logger.info("Setting up training arguments with Adafactor optimizer for maximum stability...")
    compute_metrics_with_tokenizer = partial(compute_metrics, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8, # On peut entraîner un peu plus longtemps avec un bon dataset
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        
        optim="adafactor",
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        weight_decay=0.001,
        
        fp16=False,
        
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=2,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_tokenizer,
    )

    logger.info("--- Starting training with Adafactor and clean data ---")
    trainer.train()
    logger.info("--- Training complete ---")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Best model and tokenizer saved to {output_dir}")

    logger.info("Evaluating final model...")
    final_metrics = trainer.evaluate()
    logger.info("📊 Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")

# --- 6. Point d'Entrée du Script ---
if __name__ == "__main__":
    csv_path = "Draxlmaier_QA_1000.csv"
    output_dir = "Draxlmaier_QA_1000_finetuned_lora"
    
    if not os.path.exists(csv_path):
        logger.error(f"Error: The file was not found at '{csv_path}'. Please check the path.")
    else:
        finetune_lora_from_csv(csv_path=csv_path, output_dir=output_dir)