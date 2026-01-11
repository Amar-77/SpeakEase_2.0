from transformers import AutoModelForSpeechSeq2Seq
import torch

# 1. Load your Fine-Tuned Model
# Make sure the path points to your actual folder
model_path = "./final-svarah-whisper-model" 
print(f"Loading model from: {model_path}...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)

# 2. Print the Architecture
print("\n" + "="*40)
print("   WHISPER MODEL ARCHITECTURE")
print("="*40 + "\n")

print(model)

# 3. Count the Parameters (Optional Flex)
total_params = sum(p.numel() for p in model.parameters())
print("\n" + "="*40)
print(f" TOTAL PARAMETERS: {total_params:,}")
print("="*40 + "\n")