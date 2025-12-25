# =============================================================================
# SpeakEase Backend: Handles requests from Flutter App
# =============================================================================

from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import os
import tempfile
import re
import json
import librosa
import difflib
import uvicorn
import warnings
from typing import List, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="SpeakEase Mobile Backend")

# --- 1. CONFIGURATION ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 SpeakEase Server starting... Device: {device}")

# --- 2. LOAD MODELS (Ensure these paths match your folders!) ---
print(" ⏳ Loading Transcription Model...")
transcription_pipe = pipeline("automatic-speech-recognition", model="./final-svarah-whisper-model", device=device)

print(" ⏳ Loading Scoring Model...")
scoring_model_path = "./SpeakEase_MultiScore_Model"
scoring_feature_extractor = AutoFeatureExtractor.from_pretrained(scoring_model_path)
scoring_model = AutoModelForAudioClassification.from_pretrained(scoring_model_path)
scoring_model.to(device)
with open(f"{scoring_model_path}/norm_params.json", 'r') as f:
    norm_params = json.load(f)

print(" ⏳ Loading Age Model...")
age_pipe = pipeline("audio-classification", model="./SpeakEase_Age_Checker_Model", device=device)

print("✅ All Models Loaded!")

# --- 3. HELPER FUNCTIONS ---
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)

def denormalize(value, task_name, params):
    min_val = params[task_name]['min']
    max_val = params[task_name]['max']
    value = float(value)
    value = min(max(value, 0.0), 1.0)
    return value * (max_val - min_val) + min_val

def analyze_word_by_word_advanced(reference_text: str, recognized_text: str) -> Dict[str, Any]:
    ref_clean = clean_text(reference_text)
    hyp_clean = clean_text(recognized_text)
    ref_words = ref_clean.split() if ref_clean else []
    hyp_words_clean = hyp_clean.split() if hyp_clean else []
    hyp_words_original = recognized_text.split() if recognized_text.strip() else []

    if len(ref_words) == 0:
        return {"words": [], "is_perfect_match": False, "wer": 1.0}

    # Perfect match check
    if ref_words == hyp_words_clean:
        return {
            "words": [{"text": w, "color": "green", "status": "perfect"} for w in hyp_words_original],
            "is_perfect_match": True,
            "wer": 0.0
        }

    # Diff Logic
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words_clean)
    word_results = []
    hyp_idx_counter = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for _ in range(i1, i2):
                if hyp_idx_counter < len(hyp_words_original):
                    word_results.append({"text": hyp_words_original[hyp_idx_counter], "color": "black", "status": "correct"})
                    hyp_idx_counter += 1
        elif tag == 'replace':
            for _ in range(j1, j2):
                if hyp_idx_counter < len(hyp_words_original):
                    word_results.append({"text": hyp_words_original[hyp_idx_counter], "color": "red", "status": "incorrect"})
                    hyp_idx_counter += 1
        elif tag == 'insert':
            for _ in range(j1, j2):
                if hyp_idx_counter < len(hyp_words_original):
                    word_results.append({"text": hyp_words_original[hyp_idx_counter], "color": "red", "status": "extra"})
                    hyp_idx_counter += 1
        elif tag == 'delete':
            for k in range(i1, i2):
                word_results.append({"text": ref_words[k], "color": "gray", "status": "omitted"})

    wer = 1 - matcher.ratio()
    return {"words": word_results, "is_perfect_match": False, "wer": wer}

# --- 4. THE API ENDPOINT ---
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...), correct_text: str = Form(...)):
    print(f"📥 Received file: {file.filename} | Text: {correct_text}")
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        # 1. Transcription
        transcription_output = transcription_pipe(temp_path)
        recognized_text = transcription_output.get("text", "").strip()

        # 2. Age Detection
        age_output = age_pipe(temp_path)
        
        # 3. Quality Scoring
        audio, sr = librosa.load(temp_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        inputs = scoring_feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = scoring_model(**inputs).logits.cpu().numpy()[0]

        # 4. Calculate Metrics
        fluency = denormalize(logits[0], 'fluency', norm_params) * 10
        pronunciation = denormalize(logits[1], 'pronunciation', norm_params) * 10
        accuracy_score = denormalize(logits[2], 'accuracy', norm_params) * 10
        accuracy_score = min(accuracy_score, 100) # Cap at 100

        # Word Analysis
        analysis = analyze_word_by_word_advanced(correct_text, recognized_text)
        wer_accuracy = (1 - analysis['wer']) * 100
        
        # Weighted Overall Score
        overall_score = (wer_accuracy * 0.4) + (fluency * 0.3) + (pronunciation * 0.3)
        
        # Words Per Minute
        wpm = (len(recognized_text.split()) / duration) * 60 if duration > 0 else 0

        # Response Packet for Flutter
        return {
            "speaker_analysis": {
                "predicted_age_group": age_output[0]['label'],
                "confidence": f"{age_output[0]['score']:.2f}"
            },
            "quality_scores": {
                "overall_score": f"{overall_score:.2f}",
                "fluency": f"{fluency+3:.2f}",
                "pronunciation": f"{pronunciation+3:.2f}",
                "clarity": f"{accuracy_score+2:.0f}"
            },
            "transcription_metrics": {
                "accuracy_from_wer": f"{wer_accuracy:.2f}%",
                "words_per_minute": f"{wpm:.0f}"
            },
            "full_transcription": recognized_text,
            "word_analysis": analysis['words']
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- 5. RUN SERVER ---
# This block allows you to run "python server.py" directly
if __name__ == "__main__":
    # HOST 0.0.0.0 is MANDATORY for Mobile/Emulator access
    uvicorn.run(app, host="0.0.0.0", port=8000)