import os
import glob
import json
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Load Environment Variables for local use
load_dotenv()

st.set_page_config(page_title="Sleight of Mouth (SOM) Agent", layout="wide")

# Initialize Gemini client
# It will first try st.secrets (Streamlit Cloud), then fallback to local env
try:
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API Key not found. Please set GEMINI_API_KEY in Streamlit secrets via 'Advanced settings'.")
    st.stop()

# Instantiate Google GenAI SDK Client
client = genai.Client(api_key=API_KEY)

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

st.title("🧩 Sleight of Mouth (Фокусы Языка) Gemini Agent")
st.markdown("Analyze sentences, challenge beliefs, and explore the **Alexander Gerasimov SOM** methodology.")
st.caption(f"Loaded Knowledge Sources: {len(loaded_files)} patterns")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Теперь я работаю на базе мощного **Gemini 3.1 Pro**. Напиши мне любое убеждение, и я разложу его по паттернам Фокусов Языка или предложу варианты ответа."}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Напиши убеждение для разбора (например: 'Я слишком стар для этого')..."):
    # Append User msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Gemini API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # We explicitly flatten the conversation tree to ensure perfect systemic behavior 
            # while leveraging Gemini's massive context window capabilities.
            prompt_history = system_prompt + "\n\n--- Conversation History ---\n"
            for m in st.session_state.messages:
                prompt_history += f"\n{m['role'].upper()}: {m['content']}"
            
            prompt_history += "\nASSISTANT: "

            response = client.models.generate_content_stream(
                model='gemini-1.5-pro',
                contents=prompt_history
            )

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error connecting to Gemini API: {e}")
            full_response = "Извините, произошла ошибка при обращении к API Gemini."
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
