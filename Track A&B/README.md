# Week 8: Multimodal LLM — Tracks A & B

## Overview

This part of the Week 8 Multimodal LLM lab covers two complementary tracks:

### **Track A — Voice-Interactive LLM**
Speech → Text → LLM → Speech  
Converts spoken questions into transcribed text, sends them through a local LLM (TinyLlama 1.1B-Chat), and synthesizes audio responses with gTTS.

### **Track B — Conversational Data Visualization**
Natural-language → Chart generation.  
Accepts text queries like *“Plot accuracy by game number”* and uses a validated plot-spec mapping to generate safe Matplotlib visualizations.

Both tracks extend the Week 7 model into multimodal interaction pipelines that can later be integrated into **Track C’s router** or a Streamlit frontend.

---

## 🎙️ Track A Pipeline: Speech → LLM → Speech

| Stage | Component | Description |
|--------|------------|-------------|
| 🎤 **Speech Input** | `openai-whisper` | Transcribes `.wav` audio into text. |
| 🧠 **Reasoning** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Generates concise, factual answers to transcribed queries. |
| 🔊 **Speech Output** | `gTTS` (Text-to-Speech) | Synthesizes the model’s text answer into spoken audio. |

**Example Interaction:**
- **Input (audio):** “What is the most powerful piece in chess?”
- **Output (audio):** “The queen is the most powerful piece in chess because it can move any number of squares in any direction.”

---

## 📊 Track B Pipeline: Natural Language → Matplotlib Visualization

| Stage | Component |
|--------|------------|
| 💬 **NL Prompt** | Example: “Plot move-prediction accuracy by game number.” |
| 🧩 **Spec Mapping** | Maps NL query to safe `{x, y}` plot specification. |
| 📈 **Plot Generation** | `matplotlib.pyplot` — Single-figure plots, default style (no seaborn or color customization). |


