import os
import glob
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load Environment Variables for local use
load_dotenv()

st.set_page_config(page_title="Sleight of Mouth (SOM) Agent", layout="wide")

# Initialize OpenAI client 
# It will first try st.secrets (Streamlit Cloud), then fallback to local env
try:
    API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except Exception:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("OpenAI API Key not found. Please set OPENAI_API_KEY in .env or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)

@st.cache_data
def load_som_knowledge():
    """Dynamically loads all available JSON schemas from the knowledge base."""
    # Assuming app.py is in the root directory alongside knowledge_base folder
    kb_path = os.path.join("knowledge_base", "structured", "*.json")
    files = glob.glob(kb_path)
    
    knowledge = []
    headers = []
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                knowledge.append(json.dumps(data, ensure_ascii=False))
                headers.append(os.path.basename(file_path))
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            
    return "\n---\n".join(knowledge), headers

# Load knowledge base
som_definitions, loaded_files = load_som_knowledge()

system_prompt = f"""
You are the Alexander Gerasimov Sleight of Mouth (SOM / Фокусы Языка) Agent. 
Your primary task is to analyze user input and identify which Sleight of Mouth patterns apply, 
or to demonstrate and teach these patterns appropriately.

CRITICAL RULES:
1. Primary Source: When generating content or definitions, rely strictly on the provided JSON knowledge schemas. Do not invent patterns outside of these definitions.
2. Handling Ambiguity: If a user statement fits multiple Sleight of Mouth patterns, DO NOT choose just one. Instead, assign probabilities to each possibility (summing to 100%) and present both/all of them.
3. Verification: You should explicitly reference the knowledge base elements you are drawing from to justify your logic.

KNOWLEDGE BASE JSON SOURCES:
{som_definitions}

Provide professional, accurate, and insightful analysis of text based on these SOM patterns. 
Always aim to unpack the user's beliefs or structures before applying the pattern.
"""

st.title("🧩 Sleight of Mouth (Фокусы Языка) AI Agent")
st.markdown("Analyze sentences, challenge beliefs, and explore the **Alexander Gerasimov SOM** methodology.")
st.caption(f"Loaded Knowledge Sources: {len(loaded_files)} patterns")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Привет! Напиши мне любое убеждение, и я разложу его по паттернам Фокусов Языка или предложу варианты ответа."}
    ]

# Display chat history (skipping the system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Напиши убеждение для разбора (например: 'Я слишком стар для этого')..."):
    # Append User msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call OpenAI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Note: We are using a modern model that handles large JSON structures well.
            for response in client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error connecting to OpenAI: {e}")
            full_response = "Извините, произошла ошибка при обращении к API."
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
