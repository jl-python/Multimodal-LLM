
import json, torch, platform, matplotlib.pyplot as plt, pandas as pd
from io import StringIO
from gtts import gTTS
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, os

# ------------------------------------------------
#  INIT & CONFIG
# ------------------------------------------------
def init_configs():
    run_config = {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "speech_model": "openai-whisper-base",
        "tts_engine": "gTTS",
        "visualization": "matplotlib",
        "router_keywords": {
            "speech": ["audio", "voice", "speak"],
            "viz": ["plot", "chart", "graph", "visualize"]
        }
    }
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "os": platform.system(),
        "gpu_available": torch.cuda.is_available()
    }
    with open("week8_run_config.json", "w") as f:
        json.dump(run_config, f, indent=4)
    with open("env_week8.json", "w") as f:
        json.dump(env, f, indent=4)

# ------------------------------------------------
#  LLM (TinyLlama)
# ------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model_llm = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
).to("cuda" if torch.cuda.is_available() else "cpu")

def run_llm(prompt: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer.encode(template, return_tensors="pt").to(device)
    outputs = model_llm.generate(inputs, max_new_tokens=100, do_sample=True, temperature=0.3, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def tool_llm(query: str):
    return run_llm(query)

# ------------------------------------------------
#  SPEECH (Whisper + gTTS)
# ------------------------------------------------
def tool_speech(file_path):
    import soundfile as sf
    import numpy as np
    import whisper
    import librosa

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1️⃣ Load Whisper model
    whisper_model = whisper.load_model("base")

    # 2️⃣ Load audio file
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    audio = audio.astype(np.float32)

    # 3️⃣ Transcribe audio → text
    transcript = whisper_model.transcribe(audio)["text"]

    # 4️⃣ Load LLM (TinyLlama)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # 5️⃣ Generate LLM answer
    prompt = f"User asked (via audio): {transcript}\nAnswer clearly and concisely:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs,
                             max_new_tokens=150,
                             do_sample=True,          
                             temperature=0.7,  
                             top_p=0.9)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"🎧 **Transcribed:** {transcript}\n\n🤖 **LLM Answer:** {answer}"



# ------------------------------------------------
#  VISUALIZATION (Adaptive CSV/Text Parsing)
# ------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import json, re
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load TinyLlama globally (reuse from your LLM section)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def tool_viz(query, dataset_text=None, uploaded_csv=None):
    # 1️⃣ Load dataset
    try:
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
        elif dataset_text:
            df = pd.read_csv(StringIO(dataset_text))
        else:
            return "No dataset provided."
    except Exception as e:
        return f"Error loading dataset: {e}"

    # 2️⃣ Ask LLM what to plot
    prompt = f"""
    The dataset has columns: {list(df.columns)}.
    The user asked: '{query}'.
    Suggest which column to use for x-axis and which for y-axis.
    Return only JSON as {{"x": "col_name", "y": "col_name"}}.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.6, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 3️⃣ Extract JSON safely
    try:
        match = re.search(r"\{.*\}", response)
        axes = json.loads(match.group(0)) if match else {}
        x_col, y_col = axes.get("x"), axes.get("y")
    except Exception:
        return f"Could not parse LLM output: {response}"

    # 4️⃣ Validate
    if x_col not in df.columns or y_col not in df.columns:
        # fallback to numeric columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            x_col, y_col = num_cols[:2]
        else:
            return "No valid numeric columns found."

    # 5️⃣ Plot dynamically
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    out_path = "viz_output.png"
    plt.savefig(out_path)
    plt.close()

    return out_path

