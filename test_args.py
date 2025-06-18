from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="test",
    evaluation_strategy="epoch"
)
print("OK")