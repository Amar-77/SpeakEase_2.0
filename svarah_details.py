import torch
from transformers import AutoModelForSpeechSeq2Seq

# FORCE PRINT immediately
print("1. script started...", flush=True) 

model_path = "./final-svarah-whisper-model"

print(f"2. loading from {model_path}...", flush=True)

try:
    # This is the heavy part that takes time
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    print("3. Model Loaded Successfully!", flush=True)
    
    print("\n" + "="*20, flush=True)
    print(model)  # This prints the layers
    print("="*20 + "\n", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)