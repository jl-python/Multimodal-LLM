
'''
pip install --upgrade pip
pip install -r requirements.txt
'''

import streamlit as st
from router import route_query
from tools import init_configs

st.set_page_config(page_title="Week 8 Multimodal LLM", layout="centered")
st.title("Week 8 Multimodal LLM Router")

st.markdown("Upload **audio (.wav)** or **CSV**, or type a question/dataset request below.")

# Initialize configs and environment logs
init_configs()

# --- Inputs ---
text_query = st.text_area("💬 Enter your query:")
audio_file = st.file_uploader("🎤 Upload Audio (.wav only)", type=["wav"])
csv_file = st.file_uploader("📁 Upload Dataset (CSV)", type=["csv"])
dataset_text = st.text_area("📊 Or paste CSV text here:")

if st.button("Run"):
    with st.spinner("Processing..."):
        result = route_query(
            query=text_query,
            audio_file=audio_file,
            csv_file=csv_file,
            dataset_text=dataset_text
        )

    # --- Display results dynamically ---
    if isinstance(result, str):
        if result.endswith(".png"):
            st.image(result, caption="Generated Visualization")
        elif result.endswith(".wav"):
            st.audio(result)
        else:
            st.write(result)
    elif isinstance(result, dict):
        st.json(result)
