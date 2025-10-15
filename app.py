# =============================================================================
# SpeakEase: Final Integrated 3-Model Server - REVISED FOR COLOR LOGIC
# =============================================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import os
import tempfile
import re
import json
import librosa
from typing import List, Dict, Any
import difflib
import warnings

# Suppress PySoundFile and librosa warnings
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load Deprecated as of librosa version 0.10.0.")


# --- 1. SETUP & CONFIGURATION ---
app = FastAPI(title="SpeakEase Full Analysis API")

# Helper function to get the correct device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 SpeakEase Server starting up... Using device: {device}")

# --- 2. LOAD ALL THREE SPECIALIZED MODELS AT STARTUP (PATHS ASSUMED TO BE CORRECT) ---

# --- Specialist 1: The Scribe (Transcription Model) ---
print("   - Loading Transcription Model (Whisper)...")
transcription_model_path = "./final-svarah-whisper-model"
transcription_pipe = pipeline(
    "automatic-speech-recognition",
    model=transcription_model_path,
    device=device,
)
print("   ✅ Transcription Model ready.")

# --- Specialist 2: The Quality Analyst (Multi-Score Model) ---
print("   - Loading Multi-Score Model (Wav2Vec2)...")
scoring_model_path = "./SpeakEase_MultiScore_Model"
scoring_feature_extractor = AutoFeatureExtractor.from_pretrained(scoring_model_path)
scoring_model = AutoModelForAudioClassification.from_pretrained(scoring_model_path)
scoring_model.to(device)
with open(f"{scoring_model_path}/norm_params.json", 'r') as f:
    norm_params = json.load(f)
print("   ✅ Multi-Score Model ready.")

# --- Specialist 3: The Age Detector (Child vs. Adult Classifier) ---
print("   - Loading Age Classifier Model (Wav2Vec2)...")
age_model_path = "./SpeakEase_Age_Checker_Model"
age_pipe = pipeline(
    "audio-classification",
    model=age_model_path,
    device=device,
)
print("   ✅ Age Classifier Model ready.")
print("\n🎉 All models loaded. SpeakEase server is running!")

# --- 3. HELPER & ANALYSIS FUNCTIONS ---
def clean_text(text: str) -> str:
    """Cleans text for fair comparison (lowercase, remove punctuation)."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)

def levenshtein_distance(a: List[str], b: List[str]) -> int:
    """Classic DP edit distance at word level."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def analyze_word_by_word_advanced(reference_text: str, recognized_text: str) -> Dict[str, Any]:
    """
    Updated logic for color coding:
    - Perfect Match: "green"
    - Correct Word (Non-Perfect): "black"
    - Inserted (Extra) words: "red"
    - Substituted (Misspelled) words: "red" (only the spoken word is shown)
    - Deleted (Omitted/Missing) words: "gray" (only the intended word is shown)
    """
    ref_clean = clean_text(reference_text)
    hyp_clean = clean_text(recognized_text)

    ref_words = ref_clean.split() if ref_clean else []
    hyp_words_clean = hyp_clean.split() if hyp_clean else []
    # Keep original transcribed tokens for display (preserves casing/punctuation)
    hyp_words_original = recognized_text.split() if recognized_text.strip() else []

    # Defensive check
    if len(ref_words) == 0:
        words_list = []
        for w in hyp_words_original:
            words_list.append({"text": w, "color": "red", "status": "extra"})
        return {
            "words": words_list,
            "is_perfect_match": False,
            "wer": 1.0
        }

    # Perfect match check
    is_perfect = (ref_words == hyp_words_clean and len(ref_words) == len(hyp_words_clean))

    # Compute WER
    edit_dist = levenshtein_distance(ref_words, hyp_words_clean)
    wer_value = float(edit_dist) / float(len(ref_words)) if len(ref_words) > 0 else 1.0

    word_results = []

    if is_perfect:
        # Perfect match: all words are GREEN
        for w in hyp_words_original:
            word_results.append({"text": w, "color": "green", "status": "perfect"})
        return {"words": word_results, "is_perfect_match": True, "wer": 0.0}

    # Non-perfect: align using difflib
    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_words_clean)
    opcodes = sm.get_opcodes()
    
    hyp_idx_counter = 0 
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Correct words -> black (use original hyp token)
            for ref_idx in range(i1, i2):
                display_text = hyp_words_original[hyp_idx_counter] if hyp_idx_counter < len(hyp_words_original) else ref_words[ref_idx]
                word_results.append({"text": display_text, "color": "black", "status": "correct"})
                hyp_idx_counter += 1
        
        elif tag == "replace":
            # Substitution/Misspelled word: Show the spoken word in RED. 
            # We must use a heuristic to decide the display order, we'll favor showing the spoken word 
            # and treating the intended word as an implicit omission for alignment.
            
            # This logic is tricky with difflib for display. For simplicity, we'll prioritize
            # the spoken (incorrect) word for the RED color and let the deletion be inferred, 
            # as requested, by NOT showing the gray word.
            
            # Show the spoken (substituted) word as RED.
            for hyp_idx in range(j1, j2):
                display_text = hyp_words_original[hyp_idx_counter] if hyp_idx_counter < len(hyp_words_original) else hyp_words_clean[hyp_idx]
                word_results.append({"text": display_text, "color": "red", "status": "incorrect"})
                hyp_idx_counter += 1
            
            # *CRITICAL CHANGE:* DO NOT show the intended word (ref_words[i1:i2]) as gray here. 
            # The replacement is marked entirely by the red spoken word.
            
        elif tag == "insert":
            # Extra words in hypothesis -> red
            for hyp_idx in range(j1, j2):
                display_text = hyp_words_original[hyp_idx_counter] if hyp_idx_counter < len(hyp_words_original) else hyp_words_clean[hyp_idx]
                word_results.append({"text": display_text, "color": "red", "status": "extra"})
                hyp_idx_counter += 1
                
        elif tag == "delete":
            # Missing words from hypothesis (Omission) -> show the reference word in GRAY
            for ref_idx in range(i1, i2):
                display_text = ref_words[ref_idx] 
                word_results.append({"text": display_text, "color": "gray", "status": "omitted"})

    return {
        "words": word_results,
        "is_perfect_match": is_perfect,
        "wer": wer_value
    }

def denormalize(value, task_name, params):
    """Converts a normalized 0-1 score back to the configured scale."""
    min_val = params[task_name]['min']
    max_val = params[task_name]['max']
    value = float(value)
    value = min(max(value, 0.0), 1.0)
    return value * (max_val - min_val) + min_val

def calculate_overall_score(accuracy_wer, fluency, pronunciation):
    """Calculates a weighted overall score."""
    return (accuracy_wer * 0.4) + (fluency * 0.3) + (pronunciation * 0.3)

# --- 4. API ENDPOINTS ---
@app.get("/")
async def serve_page():
    return FileResponse('index.html')

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...), correct_text: str = Form(...)):
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_path:
        temp_file_name = temp_audio_path.name
        temp_audio_path.write(await file.read())

    try:
        # === Run All Three Models ===
        transcription_output = transcription_pipe(temp_file_name)
        age_output = age_pipe(temp_file_name)

        # Prepare audio array and scoring inputs
        audio_array, sr = librosa.load(temp_file_name, sr=16000, mono=True)
        # Get Duration for WPM
        audio_duration = librosa.get_duration(y=audio_array, sr=sr) 

        inputs = scoring_feature_extractor(audio_array, sampling_rate=16000, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            scoring_logits = scoring_model(**inputs).logits.cpu().numpy()[0]

        # === Process and Combine Results ===
        recognized_text = transcription_output.get("text", "").strip()

        # Process scores
        fluency = denormalize(scoring_logits[0], 'fluency', norm_params) * 10
        pronunciation = denormalize(scoring_logits[1], 'pronunciation', norm_params) * 10
        model_accuracy = denormalize(scoring_logits[2], 'accuracy', norm_params) * 10

        if model_accuracy>100:
            model_accuracy=100

        # Process age
        predicted_age_group = age_output[0]['label']
        age_confidence = age_output[0]['score']

        # Word analysis and WER
        word_analysis = analyze_word_by_word_advanced(correct_text, recognized_text)
        wer = float(word_analysis["wer"])
        accuracy_from_wer = (1 - wer) * 100

        # WPM Calculation
        spoken_words = len(recognized_text.split())
        words_per_minute = (spoken_words / audio_duration) * 60 if audio_duration > 0 else 0

        overall_score = calculate_overall_score(accuracy_from_wer, fluency, pronunciation)

        # --- Assemble the Final, Complete Report ---
        return {
            "speaker_analysis": {
                "predicted_age_group": predicted_age_group,
                "confidence": f"{age_confidence:.2f}"
            },
            "quality_scores": {
                "overall_score": f"{overall_score:.2f}",
                "fluency": f"{fluency+5:.2f}",
                "pronunciation": f"{pronunciation+5:.2f}",
                "clarity": f"{model_accuracy+3:.0f}"
            },
            "transcription_metrics": {
                "accuracy_from_wer": f"{accuracy_from_wer:.2f}%",
                "word_error_rate": f"{wer:.2f}",
                "words_per_minute": f"{words_per_minute:.2f}"
            },
            "full_transcription": recognized_text,
            "reference_text": correct_text,
            "word_analysis": word_analysis["words"],
            "is_perfect_match": word_analysis["is_perfect_match"]
        }
    finally:
        try:
            os.remove(temp_file_name)
        except Exception:
            pass