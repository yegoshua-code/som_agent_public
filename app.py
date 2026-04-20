import os
import glob
import json
import streamlit as st
import time
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

def is_rtl(text: str) -> bool:
    """Check if text contains primarily RTL characters (Hebrew/Arabic ranges)."""
    if not text: return False
    for char in text:
        if '\u0590' <= char <= '\u08FF':
            return True
    return False

class RouterDecision(BaseModel):
    needs_research: bool
    patterns: list[str]

class SOMAnalysisItem(BaseModel):
    quote: str
    has_pattern: bool
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
1. ANALYSIS LANGUAGE: Write your analysis ('quote' and 'explanation' fields) in the EXACT SAME LANGUAGE as the user's input text (e.g., if Hebrew, explanation is in Hebrew).
2. GENERAL REPLY LANGUAGE: The 'general_reply' field MUST ALWAYS be written in English. Do not use Russian.
3. ENGLISH PATTERN NAMES: The 'som_pattern' field MUST ALWAYS be written in English (e.g., "Intention", "Meta Frame", "Change Frame Size"), regardless of the conversation language.
4. If a quote has a Sleight of Mouth pattern: set has_pattern=true, write the name of the pattern in som_pattern, and provide a clear justification in explanation.
5. If a quote is an ordinary statement, question, or fact with NO manipulative pattern: set has_pattern=false, set som_pattern to 'None' (or equivalent in target language), and briefly state why it has no pattern in the explanation.
6. IF the text requires deep explanation or context, router will provide RAW text. Use it to deepen your analysis.
7. Handling Ambiguity: If a user statement fits multiple Sleight of Mouth patterns, DO NOT choose just one. Instead, assign probabilities to each possibility (summing to 100%) and present both/all of them.
8. NO CITATIONS: Do not include source references, academic citations, or tags like [cite: ...] in your output. Provide the explanation smoothly and naturally without explicitly naming the knowledge base parts.

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
        {"role": "assistant", "content": "Hello! I am operating in fast mode on **Gemini 2.5 Flash**.\n\nEnter any single belief or paste a dialogue snippet (e.g., from a movie), and I will break it down line by line, separating ordinary facts from hidden SOM patterns!"}
    ]

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], str):
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                # Structured Rendering
                data = msg["content"]
                if data.get("general_reply"):
                    st.markdown(data["general_reply"], unsafe_allow_html=True)
                
                for i_idx, item in enumerate(data.get("items", [])):
                    with st.container():
                        is_rtl_card = is_rtl(item['quote'])
                        card_dir = "rtl" if is_rtl_card else "ltr"
                        card_align = "right" if is_rtl_card else "left"
                        border_side = "border-right" if is_rtl_card else "border-left"

                        if item.get("has_pattern", True):
                            html_card = f"""
                            <div dir="{card_dir}" style="background-color: #eef4f9; padding: 15px; border-radius: 8px; margin-bottom: 10px; {border_side}: 4px solid #bbdefb; text-align: {card_align};">
                              💡 <b>Quote:</b>
                              <div style="margin-top: 5px; margin-bottom: 15px; font-size: 16px; font-family: sans-serif;">{item['quote']}</div>
                              <b>Pattern:</b> <span dir="ltr" style="display:inline-block;">{item['som_pattern']}</span><br><br>
                              <b>Explanation:</b>
                              <div style="margin-top: 5px;">{item['explanation']}</div>
                            </div>
                            """
                            st.markdown(html_card, unsafe_allow_html=True)
                            
                            # Показываем готовую утилизацию, если она уже сгенерирована
                            if "utilization" in item:
                                util_html = item['utilization'].replace('\n', '<br>')
                                util_card = f"""
                                <div dir="{card_dir}" style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 15px; {border_side}: 5px solid #4CAF50; text-align: {card_align};">
                                  ✅ <b>Utilization ('<span dir="ltr" style="display:inline-block;">{item['som_pattern']}</span>'):</b>
                                  <div style="margin-top: 10px;">{util_html}</div>
                                </div>
                                """
                                st.markdown(util_card, unsafe_allow_html=True)
                            else:
                                ui_key = f"utilize_{idx}_{i_idx}"
                                if st.button(f"🌀 Utilize '{item['som_pattern']}'", key=f"btn_{ui_key}"):
                                    st.session_state.util_trigger = {
                                        "msg_idx": idx,
                                        "item_idx": i_idx,
                                        "quote": item['quote'],
                                        "pattern": item['som_pattern']
                                    }
                                    st.rerun()
                                    
                        else:
                            st.markdown(f"<div dir='{card_dir}' style='text-align: {card_align}; margin-bottom: 10px;'>&gt; 💬 <i>{item['quote']}</i><br>&gt; <small>ℹ️ Context: {item['explanation']}</small></div>", unsafe_allow_html=True)

# Check if we need to run an inline utilization request before handling new chat
if "util_trigger" in st.session_state:
    util_req = st.session_state.util_trigger
    del st.session_state.util_trigger
    with st.spinner(f"🌀 Generating utilization for '{util_req['pattern']}'..."):
        util_prompt = f"""Контекст:
Собеседник сказал: {util_req['quote']}
Был применен фокус: {util_req['pattern']}

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
1. Найди в "Матрице Утилизации" контр-фокусы, которые бьют паттерн '{util_req['pattern']}'.
2. Сгенерируй 3 профессиональных варианта ответа (утилизации), используя ИМЕННО ЭТИ разрешенные контр-фокусы. Для каждого варианта укажи, какой именно контр-фокус был применен (имена контр-фокусов пиши на Английском языке).
3. ОТВЕЧАЙ НА ТОМ ЖЕ ЯЗЫКЕ: Сами сгенерированные фразы обязаны соответствовать языку цитаты (Иврит, Русский, Английский).
"""
        try:
            r = client.models.generate_content(model='gemini-2.5-flash', contents=util_prompt)
            # Embed the utilization into the original message JSON object
            st.session_state.messages[util_req["msg_idx"]]["content"]["items"][util_req["item_idx"]]["utilization"] = r.text
        except Exception as e:
            st.error(f"Utilization Error: {str(e)}")
        finally:
            st.rerun()

# Processing new text prompts
prompt = st.chat_input("Enter a belief or paste a dialogue snippet for analysis...")
if prompt:
    # Append User msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Call Gemini API
    with st.chat_message("assistant"):
        try:
            # Stage 1: Fast Router (JSON only) to determine if research is needed
            router_prompt = "INSTRUCTION: You are the SOM Router. Does this user message require deep research into the raw Gerasimov books, or is it obvious enough (certainty > 45%) to answer just with the JSON schemas below? Evaluate ambiguity. If everything is clear, output needs_research=false. If probability < 45% or the topic is ambiguous, output needs_research=true and provide the list of patterns (e.g. ['01_intention', '13_apply_to_self']) to research.\n\nJSON KNOWLEDGE:\n" + som_definitions + "\n\nUSER MESSAGE:\n" + prompt
            
            with st.spinner("🔍 Analyzing (Router)..."):
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
                with st.spinner(f"📚 Researching full patterns: {', '.join(found_patterns)}... (probability < 45%)"):
                    raw_context = "\n\n--- ADDITIONAL RESEARCH EXTRACTED BY ROUTER ---\n"
                    for pattern in found_patterns:
                        raw_context += load_specific_raw_pattern(pattern)
                    
                    # Inject research right before the request
                    prompt_history = raw_context + "\n" + prompt_history

            # Final generation (Structured JSON)
            with st.spinner("🧠 Generating analysis..."):
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
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an API error occurred: {str(e)}"})

# ================================
# SIDEBAR: HTML EXPORT
# ================================

def export_chat_to_html(messages):
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>SOM Dialog Export</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; max-width: 800px; margin: 0 auto; color: #333; }
            .msg { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
            .user { background-color: #e6f3ff; border-left: 5px solid #2196F3; }
            .assistant { background-color: #f9f9f9; border-left: 5px solid #4CAF50; }
            .role { font-weight: bold; margin-bottom: 5px; color: #555; }
            .card { background: white; padding: 10px; margin-top: 10px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h2>Sleight of Mouth (SOM) Agent - Dialogue History</h2>
        <hr>
    """
    for m in messages:
        if m['role'] == "system": continue
        role_label = "User" if m['role'] == "user" else "Analyst (SOM)"
        
        # Check overall role text to set direction, but better if we check inside
        html += f'<div class="msg {m["role"]}"><div class="role">{role_label}:</div>'
        
        content = m['content']
        if isinstance(content, str):
            html += f'<div dir="auto">{content}</div>'
        else:
            html += f'<div dir="auto">{content.get("general_reply", "")}</div>'
            for item in content.get('items', []):
                card_dir = "rtl" if is_rtl(item['quote']) else "ltr"
                align = "right" if card_dir == "rtl" else "left"
                border_side = "border-right" if card_dir == "rtl" else "border-left"

                if item.get("has_pattern", True):
                    html += f'''
                    <div class="card" dir="{card_dir}" style="text-align: {align}; {border_side}: 4px solid #bbdefb;">
                        <b>Quote:</b> {item['quote']}<br>
                        <b>Pattern:</b> <span dir="ltr">{item['som_pattern']}</span><br>
                        <b>Explanation:</b> {item['explanation']}
                    </div>
                    '''
                    if "utilization" in item:
                        formatted_util = item['utilization'].replace('\n', '<br>')
                        html += f'''
                        <div class="card" dir="{card_dir}" style="background-color: #e8f5e9; {border_side}: 5px solid #4CAF50; text-align: {align};">
                            <b>✅ Utilization:</b><br>{formatted_util}
                        </div>
                        '''
                else:
                    html += f'''
                    <div dir="{card_dir}" style="margin: 10px 0; padding-{align}: 15px; {border_side}: 3px solid #ccc; color: #666; text-align: {align};">
                        <i>"{item['quote']}"</i><br>
                        <small>Context: {item['explanation']}</small>
                    </div>
                    '''
        html += '</div>'
    html += "</body></html>"
    return html

with st.sidebar:
    st.title("💾 Export Dialogue")
    st.write("You can save the entire analysis as an HTML file. It can be easily shared or printed to a perfect PDF (via 'Print' in the browser).")
    if len(st.session_state.messages) > 1:
        st.download_button(
            label="⬇️ Download Dialogue (HTML)",
            data=export_chat_to_html(st.session_state.messages),
            file_name="som_dialogue.html",
            mime="text/html"
        )

