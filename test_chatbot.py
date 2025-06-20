import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

# --- 1. Définir les chemins ---
base_model_name = "google/flan-t5-base"
# !! ATTENTION: Mettez ici le nom exact de votre dossier de sortie !!
peft_model_path = "innovatech_chatbot_final" 

# --- 2. Charger le tokenizer et le modèle ---
tokenizer = T5Tokenizer.from_pretrained(base_model_name)
base_model = T5ForConditionalGeneration.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, peft_model_path)
model = model.merge_and_unload() # Fusionner les poids pour une inférence plus rapide

# Mettre le modèle en mode évaluation
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Modèle '{peft_model_path}' chargé sur le périphérique : {device}")
print("--- Chatbot prêt. Tapez 'quitter' pour arrêter. ---")

# --- 3. Boucle d'interaction ---
while True:
    user_question = input("Vous: ")
    if user_question.lower() == 'quitter':
        break

    # Préparer l'input pour le modèle (en minuscules et avec le préfixe)
    input_text = "question: " + user_question.lower()
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Générer la réponse
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=256,
            num_beams=5, # Utiliser le "beam search" pour de meilleures réponses
            early_stopping=True,
            no_repeat_ngram_size=2 # Empêche les répétitions de paires de mots
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Chatbot: {response}")