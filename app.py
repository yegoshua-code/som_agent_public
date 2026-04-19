import os
import glob
import json
import streamlit as st
import time
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

class RouterDecision(BaseModel):
    needs_research: bool
    patterns: list[str]


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
    kb_path_json = os.path.join("knowledge_base", "structured", "*.json")
    files_json = glob.glob(kb_path_json)
    
    knowledge = []
    headers = []
    
    # Load Structured JSONs
    for file_path in files_json:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                knowledge.append(json.dumps(data, ensure_ascii=False))
                headers.append(os.path.basename(file_path))
        except Exception as e:
            st.error(f"Error loading JSON {file_path}: {e}")
            
    return "\n---\n".join(knowledge), headers

def load_specific_raw_pattern(pattern_prefix: str) -> str:
    """Loads a specific raw .txt file if requested by the Router."""
    kb_path_raw = os.path.join("knowledge_base", "raw", f"{pattern_prefix}*")
    files_raw = glob.glob(kb_path_raw)
    if not files_raw:
        return ""
    try:
        with open(files_raw[0], "r", encoding="utf-8") as f:
            return f"RAW TEXT RESOURCE ({os.path.basename(files_raw[0])}):\n{f.read()}\n"
    except:
        return ""

# Load knowledge base
som_definitions, loaded_files = load_som_knowledge()

system_prompt = f"""
You are the Alexander Gerasimov Sleight of Mouth (SOM / Фокусы Языка) Agent. 
Your primary task is to analyze user input and identify which Sleight of Mouth patterns apply, 
or to demonstrate and teach these patterns appropriately.

CRITICAL RULES:
1. Primary Source: When generating content or definitions, rely heavily on the provided JSON knowledge schemas. If the boundaries of a pattern are unclear or if there is a `[cite: X]` reference in the JSON, search the RAW TEXT RESOURCES included below to provide exact theory and nuances from Alexander Gerasimov.
2. Handling Ambiguity: If a user statement fits multiple Sleight of Mouth patterns, DO NOT choose just one. Instead, assign probabilities to each possibility (summing to 100%) and present both/all of them.
3. Dialogue Analysis (Разбор диалогов): Если пользователь отправляет длинный текст, диалог или сцену, проанализируй текст структурно по репликам. Четко отделяй обычные факты (перечисление, вопросы, искренние эмоции) от применения Фокусов Языка. Объясняй логику и приводящие к фокусу убеждения.
4. Verification: You should explicitly reference the knowledge base elements and citations you are drawing from to justify your logic.

KNOWLEDGE BASE (JSON AND RAW TEXT SOURCES):
{som_definitions}

Provide professional, accurate, and insightful analysis of text based on these SOM patterns. 
Always aim to unpack the user's beliefs or structures before applying the pattern.
"""

st.title("🧩 Sleight of Mouth (Фокусы Языка) Gemini Agent")
st.markdown("Analyze sentences, challenge beliefs, and explore the **Alexander Gerasimov SOM** methodology.")
st.caption(f"Loaded Knowledge Sources: {len(loaded_files)} patterns")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Привет! Я работаю на базе мощного **Gemini 3.1 Pro**.\n\nНапиши мне любое одиночное убеждение для отработки Фокусов Языка ИЛИ отправь мне отрывок диалога (например, из фильма), и я разберу его по репликам, отделив обычные факты от скрытых манипуляций!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Напиши убеждение или вставь отрывок диалога для разбора..."):
    # Append User msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Gemini API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stage 1: Fast Router (JSON only) to determine if research is needed
            router_prompt = "INSTRUCTION: You are the SOM Router. Does this user message require deep research into the raw Gerasimov books, or is it obvious enough (certainty > 45%) to answer just with the JSON schemas below? Evaluate ambiguity. If everything is clear, output needs_research=false. If probability < 45% or the topic is ambiguous, output needs_research=true and provide the list of patterns (e.g. ['01_intention', '13_apply_to_self']) to research.\n\nJSON KNOWLEDGE:\n" + som_definitions + "\n\nUSER MESSAGE:\n" + prompt
            
            with st.spinner("🔍 Анализ (Роутер)..."):
                router_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=router_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RouterDecision,
                    ),
                )
                
            decision = json.loads(router_response.text)
            
            # Stage 2: Prepare Full Prompt
            prompt_history = system_prompt + "\n\n--- Conversation History ---\n"
            for m in st.session_state.messages:
                prompt_history += f"\n{m['role'].upper()}: {m['content']}"
            
            prompt_history += "\nASSISTANT: "
            
            if decision.get("needs_research") and decision.get("patterns"):
                found_patterns = decision.get("patterns")
                with st.spinner(f"📚 Иду изучать полные книги: {', '.join(found_patterns)}... (вероятность < 45%)"):
                    raw_context = "\n\n--- ADDITIONAL RESEARCH EXTRACTED BY ROUTER ---\n"
                    for pattern in found_patterns:
                        raw_context += load_specific_raw_pattern(pattern)
                    
                    # Inject research right before the request
                    prompt_history = raw_context + "\n" + prompt_history

            # Final generation
            response = client.models.generate_content_stream(
                model='gemini-2.5-flash',
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
