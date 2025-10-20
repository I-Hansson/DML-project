import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import os

# --- setup ---
model_name = "gpt2"                # same base model used for fine-tuning
weights_path = "./models/best_gpt2_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- reload base architecture and tokenizer ---
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

print("✅ Model weights loaded successfully")

# --- inference prompt ---
prompt = "Genre: Indie\n\nLyrics:"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# --- generate text ---
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=200,
        top_p=0.9,
        top_k=50,
        temperature=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=2,
    )

# --- decode & display ---
for i, output in enumerate(outputs, 1):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"SAMPLE {i}")
    print(f"{'='*60}")
    print(text)
