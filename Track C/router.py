
import re
from tools import tool_llm, tool_speech, tool_viz

def route_query(query="", audio_file=None, csv_file=None, dataset_text=None):
    text = (query or "").lower()

    # --- Guardrails (simple keyword blocklist) ---
    banned = ["os.", "system(", "delete", "exec(", "import os", "subprocess"]
    if any(bad in text for bad in banned):
        return " Unsafe command blocked."

    # --- Route by intent ---
    if audio_file is not None:
        return tool_speech(audio_file)
    elif re.search(r"(plot|chart|graph|visualize|trend)", text):
        return tool_viz(query, dataset_text, csv_file)
    else:
        return tool_llm(query)

