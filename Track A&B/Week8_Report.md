
# Week 8 — Multimodal Integration with LLMs, Speech, and Visualization

### Overview
This week’s project extends the Week 7 Stable Diffusion + LoRA training pipeline into a multimodal assistant that combines text reasoning, speech interaction, and visual analytics inside a unified system.  
The objective was to demonstrate that models fine-tuned for generative vision tasks (from Week 7) can now be connected to language-based reasoning interfaces using a router architecture.

### System Design
The notebook prototypes were modularized into three tracks:

1. **Baseline Text QA** – TinyLlama 1.1B handles natural-language questions related to chess tactics.  
2. **Speech LLM Pipeline** – converts user audio (.wav) → text via SpeechRecognition, routes the transcript to TinyLlama, and replies through gTTS.  
3. **Visualization Module** – interprets natural-language chart queries such as *“Plot move-prediction accuracy by game number”* and produces Matplotlib charts from a small chess-training dataset.

In this notebook (Week 8 - Track A&B) we explore these tools (pre-router) and collect latency & accuracy metrics for an ablation table.  

### Connection to Week 7 Work
Week 7 focused on image generation and adaptation through LoRA-weighted Stable Diffusion.  
The Week 8 router concept extends that work by showing how the same back-end models can be wrapped into **user-facing multimodal endpoints**.  
Where Week 7 used FastAPI to expose diffusion inference services, Week 8 - Track C uses a planned Streamlit front-end to unify all three modalities:  
- Text QA serves as the reasoning layer.  
- Speech I/O offers natural conversation.  
- Visualization produces training or evaluation charts for diffusion results.  

### Outcomes and Usefulness
The final ablation results show minimal latency increase when adding speech or visualization modules while maintaining high semantic accuracy. There are significant increases to accuracy when adding speech and visualization as tools.   
This multimodal framework lays the groundwork for an interactive AI studio capable of both generating and explaining images, graphs, and spoken responses—making Week 7’s generative models accessible through conversational interfaces.

