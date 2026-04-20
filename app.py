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

class SOMAnalysisItem(BaseModel):
    quote: str
    som_pattern: str
    explanation: str

class FullAnalysis(BaseModel):
    general_reply: str
    items: list[SOMAnalysisItem]


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

system_prompt = f"""You are an elite expert in NLP and Alexander Gerasimov's "Sleight of Mouth" (Фокусы Языка) methodology.
Your task is to analyze dialogues or sentences and identify the exact SOM patterns being used.

CRITICAL RULES:
1. MATCH THE LANGUAGE: You MUST write your analysis and explanations in the EXACT SAME LANGUAGE as the user's input text (e.g., if the user provides English text, your 'quote', 'som_pattern', and 'explanation' must be in English. If Russian, then Russian).
2. ONLY output data for quotes that explicitly contain a Sleight of Mouth pattern.
3. IGNORE ordinary questions, simple statements of fact, or pure emotional expressions with no manipulative/belief-shifting structure.
4. For each identified pattern, provide a clear, concise justification based on the rules found in your JSON knowledge base.
5. IF the text requires deep explanation or context, router will provide RAW text. Use it to deepen your analysis.
6. Handling Ambiguity: If a user statement fits multiple Sleight of Mouth patterns, DO NOT choose just one. Instead, assign probabilities to each possibility (summing to 100%) and present both/all of them.
7. Verification: You should explicitly reference the knowledge base elements and citations you are drawing from to justify your logic.

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
        {"role": "assistant", "content": "Привет! Я работаю в экономном и быстром режиме на базе **Gemini 2.5 Flash**.\n\nНапиши мне любое одиночное убеждение для отработки Фокусов Языка ИЛИ отправь мне отрывок диалога (например, из фильма), и я разберу его по репликам, отделив обычные факты от скрытых манипуляций!"}
    ]

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"])
            else:
                # Structured Rendering
                data = msg["content"]
                if data.get("general_reply"):
                    st.markdown(data["general_reply"])
                
                for i_idx, item in enumerate(data.get("items", [])):
                    with st.container():
                        st.info(f"**Цитата:** {item['quote']}\n\n**Фокус:** {item['som_pattern']}\n\n**Разбор:** {item['explanation']}")
                        
                        ui_key = f"utilize_{idx}_{i_idx}"
                        
                        if st.button(f"🌀 Утилизировать '{item['som_pattern']}'", key=f"btn_{ui_key}"):
                            st.session_state[f"show_util_{ui_key}"] = True
                            
                        # Show utilization if button was clicked
                        if st.session_state.get(f"show_util_{ui_key}"):
                            if f"util_text_{ui_key}" not in st.session_state:
                                with st.spinner("Создаю контр-ответы..."):
                                    try:
                                        util_prompt = f"""Контекст:
Собеседник сказал: {item['quote']}
Был применен фокус: {item['som_pattern']}

МАТРИЦА УТИЛИЗАЦИИ ГЕРАСИМОВА (Строгие правила контр-приемов):
- Намерение утилизируется через: Переопределение, Другой результат, Иерархия критериев, Метафрейм.
- Переопределение: Обратное переопределение, Метафрейм.
- Последствия: Последствия (встречные), Метафрейм.
- Разделение: Разделение, Обобщение, Иерархия критериев, Метафрейм.
- Обобщение: Разделение, Стратегия реальности, Противоположный пример, Метафрейм.
- Аналогия: встречная Аналогия, Стратегия реальности, Метафрейм.
- Размер фрейма: встречный Размер фрейма, Метафрейм.
- Другой результат: Последствия, встречный Другой результат, Метафрейм.
- Модель мира: контр-Модель мира, Иерархия критериев, Метафрейм.
- Стратегия реальности: Метафрейм.
- Противоположный пример: Последствия, Обобщение, Метафрейм.
- Иерархия критериев: Разделение, Стратегия реальности, Метафрейм.
- Применение к себе: Размер фрейма, Метафрейм.
- Метафрейм: Метафрейм более высокого уровня.

Задача:
1. Найди в "Матрице Утилизации" контр-фокусы, которые бьют паттерн '{item['som_pattern']}'.
2. Сгенерируй 3 профессиональных варианта ответа (утилизации), используя ИМЕННО ЭТИ разрешенные контр-фокусы. Для каждого варианта укажи, какой именно контр-фокус был применен.
3. Отвечай СТРОГО на том же языке, на котором написана цитата (если цитата на английском - утилизация на английском).
"""
                                        r = client.models.generate_content(model='gemini-2.5-flash', contents=util_prompt)
                                        st.session_state[f"util_text_{ui_key}"] = r.text
                                    except Exception as e:
                                        st.session_state[f"util_text_{ui_key}"] = "Ошибка генерации утилизации."
                            st.success(st.session_state[f"util_text_{ui_key}"])

if prompt := st.chat_input("Напиши убеждение или вставь отрывок диалога для разбора..."):
    # Append User msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Gemini API
    with st.chat_message("assistant"):
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
                if isinstance(m["content"], str):
                    prompt_history += f"\n{m['role'].upper()}: {m['content']}"
                else:
                    prompt_history += f"\n{m['role'].upper()}: [Structured SOM Analysis attached]"
            
            prompt_history += "\nASSISTANT: "
            
            if decision.get("needs_research") and decision.get("patterns"):
                found_patterns = decision.get("patterns")
                with st.spinner(f"📚 Иду изучать полные книги: {', '.join(found_patterns)}... (вероятность < 45%)"):
                    raw_context = "\n\n--- ADDITIONAL RESEARCH EXTRACTED BY ROUTER ---\n"
                    for pattern in found_patterns:
                        raw_context += load_specific_raw_pattern(pattern)
                    
                    # Inject research right before the request
                    prompt_history = raw_context + "\n" + prompt_history

            # Final generation (Structured JSON)
            with st.spinner("🧠 Генерирую разбор..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_history,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=FullAnalysis,
                    )
                )

            analysis_data = json.loads(response.text)
            
            # Append structured dict to messages instead of text
            st.session_state.messages.append({"role": "assistant", "content": analysis_data})
            st.rerun() # Rerun to render the new structured message
            
        except Exception as e:
            st.error(f"Error connecting to Gemini API: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Извините, произошла ошибка при обращении к API Gemini."})

