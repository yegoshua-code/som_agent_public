import os
import glob
import json
import streamlit as st
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ─── RTL DETECTION ────────────────────────────────────────────────────────────
def is_rtl(text: str) -> bool:
    """Returns True if text contains Hebrew or Arabic characters."""
    if not text:
        return False
    for char in text:
        if '\u0590' <= char <= '\u08FF':
            return True
    return False

# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────
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

class DraftEvaluation(BaseModel):
    detected_pattern: str
    effectiveness: str          # "Strong" / "Moderate" / "Weak"
    reasoning: str
    improvement: str

class SimulatorJudgement(BaseModel):
    target_pattern: str
    user_pattern: str
    hit: bool
    score: int                  # 0-100
    feedback: str
    expert_example: str

# ─── ENVIRONMENT & CLIENT ─────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="Sleight of Mouth Agent", layout="wide", initial_sidebar_state="expanded")

try:
    API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
except Exception:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("Gemini API Key not found. Set GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ─── KNOWLEDGE BASE ───────────────────────────────────────────────────────────
@st.cache_data
def load_som_knowledge():
    kb_path_json = os.path.join("knowledge_base", "structured", "*.json")
    files_json = glob.glob(kb_path_json)
    knowledge, headers = [], []
    for file_path in files_json:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                knowledge.append(json.dumps(data, ensure_ascii=False))
                headers.append(os.path.basename(file_path))
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return "\n---\n".join(knowledge), headers

def load_specific_raw_pattern(pattern_prefix: str) -> str:
    kb_path_raw = os.path.join("knowledge_base", "raw", f"{pattern_prefix}*")
    files_raw = glob.glob(kb_path_raw)
    if not files_raw:
        return ""
    try:
        with open(files_raw[0], "r", encoding="utf-8") as f:
            return f"RAW TEXT RESOURCE ({os.path.basename(files_raw[0])}):\n{f.read()}\n"
    except:
        return ""

som_definitions, loaded_files = load_som_knowledge()

# ─── SYSTEM PROMPTS ───────────────────────────────────────────────────────────
ANALYSIS_SYSTEM_PROMPT = f"""You are an elite expert in NLP and Alexander Gerasimov's "Sleight of Mouth" methodology.
Your task is to analyze dialogues or sentences and identify the exact SOM patterns being used.

CRITICAL RULES:
1. ANALYSIS LANGUAGE: The 'quote' and 'explanation' fields MUST be in the EXACT SAME LANGUAGE as the input text.
2. GENERAL REPLY: The 'general_reply' field MUST ALWAYS be written in English only.
3. PATTERN NAMES: The 'som_pattern' field MUST ALWAYS be in English (e.g., "Intention", "Meta Frame").
4. If a quote has a SOM pattern: set has_pattern=true, name it in som_pattern, justify in explanation.
5. If a quote is plain (no manipulation): set has_pattern=false, som_pattern='None', briefly explain why.
6. AMBIGUITY: If multiple patterns fit, assign probabilities summing to 100% and present all.
7. NO CITATIONS: Never include [cite: ...] or academic tags.

KNOWLEDGE BASE:
{som_definitions}
"""

UTILIZATION_MATRIX = """
GERASIMOV'S UTILIZATION MATRIX:
- Intention → Redefining, Another Outcome, Hierarchy of Criteria, Meta Frame
- Redefining → Counter-Redefining, Meta Frame
- Consequences → Counter-Consequences, Meta Frame
- Chunking Down → Chunking Down, Chunking Up, Hierarchy of Criteria, Meta Frame
- Chunking Up → Chunking Down, Strategy of Reality, Counter Example, Meta Frame
- Analogy → Counter-Analogy, Strategy of Reality, Meta Frame
- Change Frame Size → Counter-Change Frame Size, Meta Frame
- Another Outcome → Consequences, Counter-Another Outcome, Meta Frame
- Model of the World → Counter-Model of the World, Hierarchy of Criteria, Meta Frame
- Strategy of Reality → Meta Frame
- Counter Example → Consequences, Chunking Up, Meta Frame
- Hierarchy of Criteria → Chunking Down, Strategy of Reality, Meta Frame
- Apply to Self → Change Frame Size, Meta Frame
- Meta Frame → Higher-level Meta Frame
"""

# ─── RENDERING HELPERS ────────────────────────────────────────────────────────
def render_som_card(item: dict, msg_idx: int, i_idx: int):
    """Renders a single SOM analysis card with RTL awareness."""
    rtl = is_rtl(item['quote'])
    d = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    border = "border-right" if rtl else "border-left"

    if item.get("has_pattern", True):
        st.markdown(f"""
        <div dir="{d}" style="background-color:#eef4f9;color:#1e1e1e;padding:12px;
             border-radius:8px;margin-bottom:8px;{border}:4px solid #bbdefb;text-align:{align};">
          💡 <b>Quote:</b>
          <div style="margin:6px 0 12px;font-size:16px;">{item['quote']}</div>
          <b>Pattern:</b> <span dir="ltr" style="display:inline-block;">{item['som_pattern']}</span><br><br>
          <b>Explanation:</b>
          <div style="margin-top:4px;">{item['explanation']}</div>
        </div>""", unsafe_allow_html=True)

        if "utilization" in item:
            util_html = item['utilization'].replace('\n', '<br>')
            st.markdown(f"""
            <div dir="{d}" style="background-color:#e8f5e9;color:#1e1e1e;padding:12px;
                 border-radius:8px;margin-bottom:14px;{border}:5px solid #4CAF50;text-align:{align};">
              ✅ <b>Utilization (<span dir="ltr">{item['som_pattern']}</span>):</b>
              <div style="margin-top:8px;">{util_html}</div>
            </div>""", unsafe_allow_html=True)
        else:
            ui_key = f"btn_utilize_{msg_idx}_{i_idx}"
            if st.button(f"🌀 Utilize '{item['som_pattern']}'", key=ui_key):
                st.session_state.util_trigger = {
                    "msg_idx": msg_idx, "item_idx": i_idx,
                    "quote": item['quote'], "pattern": item['som_pattern']
                }
                st.rerun()
    else:
        st.markdown(
            f"<div dir='{d}' style='text-align:{align};margin-bottom:8px;color:#555;'>"
            f"💬 <i>{item['quote']}</i><br>"
            f"<small>ℹ️ Context: {item['explanation']}</small></div>",
            unsafe_allow_html=True
        )

def render_chat_history(messages_key: str):
    """Renders the full chat history for a given state key."""
    for idx, msg in enumerate(st.session_state[messages_key]):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, str):
                st.markdown(content, unsafe_allow_html=True)
            else:
                if content.get("general_reply"):
                    st.markdown(content["general_reply"], unsafe_allow_html=True)
                for i_idx, item in enumerate(content.get("items", [])):
                    render_som_card(item, idx, i_idx)

def run_utilization(util_req: dict, messages_key: str):
    """Calls the API to generate utilization and embeds it in history."""
    with st.spinner(f"🌀 Generating utilization for '{util_req['pattern']}'..."):
        prompt = f"""Context:
User Quote: {util_req['quote']}
Applied Pattern: {util_req['pattern']}

{UTILIZATION_MATRIX}

Task:
1. Find counter-patterns from the Utilization Matrix that beat '{util_req['pattern']}'.
2. Generate 3 professional counter-responses (utilizations) using EXACTLY these allowed counter-patterns.
3. LANGUAGE RULE: The counter-responses AND their explanations MUST be in the EXACT SAME LANGUAGE as the User Quote.
   The ONLY thing that must be in English is the counter-pattern name (e.g., "Meta Frame").
"""
        try:
            r = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            st.session_state[messages_key][util_req["msg_idx"]]["content"]["items"][util_req["item_idx"]]["utilization"] = r.text
        except Exception as e:
            st.error(f"Utilization Error: {e}")
        finally:
            st.rerun()

def run_analysis(prompt: str, messages_key: str):
    """Runs the 2-stage Router → Generator pipeline and appends result."""
    st.session_state[messages_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            router_prompt = (
                "INSTRUCTION: You are the SOM Router. Does this message require deep research (certainty < 45%)? "
                "If clear, output needs_research=false. Otherwise needs_research=true with pattern list.\n\n"
                f"JSON KNOWLEDGE:\n{som_definitions}\n\nUSER MESSAGE:\n{prompt}"
            )
            with st.spinner("🔍 Analyzing..."):
                router_resp = client.models.generate_content(
                    model='gemini-2.5-flash', contents=router_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=RouterDecision)
                )
            decision = json.loads(router_resp.text)

            history = ANALYSIS_SYSTEM_PROMPT + "\n\n--- Conversation History ---\n"
            for m in st.session_state[messages_key]:
                if isinstance(m["content"], str):
                    history += f"\n{m['role'].upper()}: {m['content']}"
                else:
                    history += f"\n{m['role'].upper()}: [Structured SOM Analysis]"
            history += "\nASSISTANT: "

            if decision.get("needs_research") and decision.get("patterns"):
                with st.spinner(f"📚 Deep-researching patterns: {', '.join(decision['patterns'])}..."):
                    for p in decision["patterns"]:
                        history = load_specific_raw_pattern(p) + "\n" + history

            with st.spinner("🧠 Generating analysis..."):
                response = client.models.generate_content(
                    model='gemini-2.5-flash', contents=history,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=FullAnalysis)
                )

            analysis_data = json.loads(response.text)
            if analysis_data.get("general_reply"):
                st.markdown(analysis_data["general_reply"], unsafe_allow_html=True)
            for i_idx, item in enumerate(analysis_data.get("items", [])):
                render_som_card(item, len(st.session_state[messages_key]) - 1, i_idx)

            st.session_state[messages_key].append({"role": "assistant", "content": analysis_data})
            st.rerun()

        except Exception as e:
            err = f"Sorry, an API error occurred: {e}"
            st.error(err)
            st.session_state[messages_key].append({"role": "assistant", "content": err})

# ─── HTML EXPORT ──────────────────────────────────────────────────────────────
def export_chat_to_html(messages: list) -> str:
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>SOM Agent - Dialogue Export</title>
    <style>
        body{font-family:sans-serif;max-width:900px;margin:40px auto;padding:20px;background:#f9f9f9;}
        .msg{margin-bottom:24px;padding:16px;border-radius:10px;}
        .user{background:#e3f2fd;} .assistant{background:#fff;border:1px solid #ddd;}
        .role{font-weight:bold;margin-bottom:8px;color:#555;}
        .card{background:#eef4f9;padding:12px;border-radius:8px;margin-top:8px;
              border-left:4px solid #bbdefb;color:#1e1e1e;}
    </style></head><body>
    <h2>Sleight of Mouth (SOM) Agent — Dialogue Export</h2><hr>"""

    for m in messages:
        if m['role'] == "system":
            continue
        role_label = "User" if m['role'] == "user" else "Analyst (SOM)"
        html += f'<div class="msg {m["role"]}"><div class="role">{role_label}:</div>'
        content = m['content']
        if isinstance(content, str):
            html += f'<div dir="auto">{content}</div>'
        else:
            html += f'<div dir="auto">{content.get("general_reply", "")}</div>'
            for item in content.get("items", []):
                d = "rtl" if is_rtl(item['quote']) else "ltr"
                al = "right" if d == "rtl" else "left"
                bs = "border-right" if d == "rtl" else "border-left"
                if item.get("has_pattern", True):
                    html += f'''<div class="card" dir="{d}" style="text-align:{al};{bs}:4px solid #bbdefb;">
                        <b>Quote:</b> {item['quote']}<br>
                        <b>Pattern:</b> <span dir="ltr">{item['som_pattern']}</span><br>
                        <b>Explanation:</b> {item['explanation']}</div>'''
                    if "utilization" in item:
                        html += f'''<div class="card" dir="{d}" style="background:#e8f5e9;{bs}:5px solid #4CAF50;text-align:{al};">
                            <b>✅ Utilization:</b><br>{item["utilization"].replace(chr(10),"<br>")}</div>'''
                else:
                    html += f'''<div dir="{d}" style="margin:8px 0;padding-{al}:12px;{bs}:3px solid #ccc;color:#666;text-align:{al};">
                        <i>"{item['quote']}"</i><br><small>Context: {item['explanation']}</small></div>'''
        html += '</div>'
    html += "</body></html>"
    return html

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR: NAVIGATION + EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧩 SOM Agent")
    app_mode = st.radio(
        "Navigate:",
        ["🗣️ Live Assistant", "🏋️ Training Simulator", "📁 File Analysis"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Knowledge Base: {len(loaded_files)} patterns loaded")

    # Export button (only for Live Assistant history)
    if app_mode == "🗣️ Live Assistant":
        st.subheader("💾 Export")
        msgs = st.session_state.get("live_messages", [])
        if len(msgs) > 1:
            st.download_button(
                label="⬇️ Download Dialogue (HTML)",
                data=export_chat_to_html(msgs),
                file_name="som_dialogue.html",
                mime="text/html"
            )
        else:
            st.caption("Start a conversation to enable export.")

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 1: LIVE ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "🗣️ Live Assistant":
    st.title("🗣️ Live Negotiation Assistant")
    st.markdown("Paste your opponent's text **or your own draft response** for analysis and counter-strategies.")

    if "live_messages" not in st.session_state:
        st.session_state.live_messages = [
            {"role": "assistant", "content":
             "Hello! I am your **Sleight of Mouth Negotiation Assistant**, powered by Gemini 2.5 Flash.\n\n"
             "**How to use:**\n"
             "- Paste your **opponent's phrases** → I'll identify hidden SOM patterns and suggest 3 counter-responses.\n"
             "- Paste **your own draft reply** → I'll evaluate the SOM pattern you applied and suggest improvements."}
        ]

    # Handle utilization trigger
    if "util_trigger" in st.session_state:
        req = st.session_state.pop("util_trigger")
        run_utilization(req, "live_messages")

    render_chat_history("live_messages")

    col1, col2 = st.columns([3, 1])
    with col1:
        live_prompt = st.chat_input("Paste opponent's text or your own draft for analysis...")
    with col2:
        analyze_own = st.toggle("📝 My own reply", key="own_draft_toggle",
                                help="Enable this if you're submitting YOUR OWN draft reply for evaluation instead of the opponent's text.")

    if live_prompt:
        if analyze_own:
            # Draft evaluation mode
            eval_prompt = f"""You are evaluating the user's own reply in a negotiation context.
The user wrote this as their draft response: "{live_prompt}"

Evaluate it using Gerasimov's Sleight of Mouth framework:
1. Which SOM pattern did the user apply (if any)?
2. How effective is it as a persuasion/counter technique (Strong / Moderate / Weak)?
3. Why — reasoning in the SAME LANGUAGE as the user's text.
4. How can they improve or strengthen it?

Return JSON matching the DraftEvaluation schema.
LANGUAGE RULE: 'reasoning' and 'improvement' fields must be in the same language as the input. Pattern names in English.
"""
            st.session_state.live_messages.append({"role": "user", "content": f"📝 **My draft reply:** {live_prompt}"})
            with st.chat_message("user"):
                st.markdown(f"📝 **My draft reply:** {live_prompt}", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("📝 Evaluating your draft..."):
                    try:
                        r = client.models.generate_content(
                            model='gemini-2.5-flash', contents=eval_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=DraftEvaluation)
                        )
                        ev = json.loads(r.text)
                        ev_html = f"""
                        <div style="background:#fff8e1;color:#1e1e1e;padding:14px;border-radius:8px;border-left:5px solid #FFC107;">
                          📝 <b>Draft Evaluation</b><br><br>
                          <b>Pattern detected:</b> <span style="font-family:monospace;">{ev.get('detected_pattern','—')}</span><br>
                          <b>Effectiveness:</b> {ev.get('effectiveness','—')}<br><br>
                          <b>Reasoning:</b><br>{ev.get('reasoning','')}<br><br>
                          <b>💡 How to improve:</b><br>{ev.get('improvement','')}
                        </div>"""
                        st.markdown(ev_html, unsafe_allow_html=True)
                        st.session_state.live_messages.append({"role": "assistant", "content": {"general_reply": "", "items": [], "_eval": ev}})
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")
        else:
            run_analysis(live_prompt, "live_messages")

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 2: TRAINING SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🏋️ Training Simulator":
    st.title("🏋️ SOM Training Simulator")
    st.markdown("Practice countering beliefs and objections. The AI will **judge your response** against Gerasimov's Matrix.")

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = {"scenario": None, "target_pattern": None, "judgement": None}

    sim = st.session_state.sim_state

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        if st.button("🎲 Generate New Objection", type="primary"):
            with st.spinner("Generating a challenging belief..."):
                gen_prompt = """Generate a realistic, challenging business or personal objection / limiting belief that a negotiator might face.
Return ONLY:
- The objection as a short natural-language sentence (max 2 sentences).
- The ONE Sleight of Mouth pattern most prominently embedded in it (English name).
Format: JSON with keys "objection" (string) and "embedded_pattern" (string)."""
                try:
                    r = client.models.generate_content(
                        model='gemini-2.5-flash', contents=gen_prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    data = json.loads(r.text)
                    st.session_state.sim_state = {
                        "scenario": data.get("objection", ""),
                        "target_pattern": data.get("embedded_pattern", ""),
                        "judgement": None
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating scenario: {e}")

    with col_reset:
        if st.button("🔄 Reset"):
            st.session_state.sim_state = {"scenario": None, "target_pattern": None, "judgement": None}
            st.rerun()

    sim = st.session_state.sim_state

    if sim["scenario"]:
        st.divider()
        st.markdown(f"""
        <div style="background:#f3e5f5;color:#1e1e1e;padding:16px;border-radius:10px;border-left:5px solid #9C27B0;font-size:17px;">
          🎯 <b>Objection to counter:</b><br><br>
          <i>"{sim['scenario']}"</i>
        </div>""", unsafe_allow_html=True)
        st.caption(f"💡 Embedded pattern hint: **{sim['target_pattern']}** — try to beat it with the right counter-pattern from Gerasimov's Matrix!")

        user_answer = st.text_area("✍️ Your counter-response:", height=100, key="sim_answer",
                                   placeholder="Write your response using a Sleight of Mouth pattern...")

        if st.button("⚖️ Judge My Response", type="primary", disabled=not user_answer.strip()):
            with st.spinner("🧑‍⚖️ Evaluating your response..."):
                judge_prompt = f"""You are an expert judge evaluating a Sleight of Mouth counter-response.

Original objection: "{sim['scenario']}"
Embedded (target) pattern in the objection: {sim['target_pattern']}

User's counter-response: "{user_answer}"

{UTILIZATION_MATRIX}

Evaluate:
1. Which SOM pattern did the user apply?
2. Is it an allowed counter-pattern from the Utilization Matrix against '{sim['target_pattern']}'?
3. Score the response 0-100 for persuasiveness and correctness.
4. Provide brief feedback in English.
5. Show an expert ideal example counter-response using the most effective allowed pattern.

Return JSON matching: target_pattern, user_pattern, hit (bool), score (int), feedback, expert_example."""
                try:
                    r = client.models.generate_content(
                        model='gemini-2.5-flash', contents=judge_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SimulatorJudgement)
                    )
                    st.session_state.sim_state["judgement"] = json.loads(r.text)
                    st.rerun()
                except Exception as e:
                    st.error(f"Judgement error: {e}")

        if sim.get("judgement"):
            j = sim["judgement"]
            hit_icon = "✅" if j.get("hit") else "❌"
            score = j.get("score", 0)
            score_color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"
            st.divider()
            st.markdown(f"""
            <div style="background:#e8f5e9;color:#1e1e1e;padding:16px;border-radius:10px;border-left:5px solid #4CAF50;">
              <b>⚖️ Judgement</b><br><br>
              {hit_icon} <b>Your pattern:</b> {j.get('user_pattern','—')}<br>
              {'✅ Valid counter-pattern!' if j.get('hit') else '❌ Not in allowed counter-patterns for this objection.'}<br><br>
              <b>Score:</b> <span style="color:{score_color};font-size:22px;font-weight:bold;">{score}/100</span><br><br>
              <b>Feedback:</b> {j.get('feedback','')}<br><br>
              <b>💡 Expert Example:</b><br><i>{j.get('expert_example','')}</i>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 3: FILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "📁 File Analysis":
    st.title("📁 Bulk File Analysis")
    st.markdown("Upload a script, speech, or conversation file. The AI will extract all SOM patterns used by the author.")

    uploaded_file = st.file_uploader(
        "Choose a file to analyze:",
        type=["txt", "pdf", "docx"],
        help="Supported formats: .txt, .pdf, .docx"
    )

    def extract_text(file) -> str:
        name = file.name.lower()
        if name.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(file)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                st.error("pypdf not installed. Add 'pypdf' to requirements.txt.")
                return ""
        elif name.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(file)
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                st.error("python-docx not installed. Add 'python-docx' to requirements.txt.")
                return ""
        return ""

    if uploaded_file:
        with st.spinner(f"Reading '{uploaded_file.name}'..."):
            raw_text = extract_text(uploaded_file)

        if raw_text.strip():
            st.success(f"✅ File loaded: {len(raw_text)} characters")
            with st.expander("📄 Preview (first 2000 characters)"):
                st.text(raw_text[:2000])

            max_chars = 12000
            if len(raw_text) > max_chars:
                st.warning(f"File is large ({len(raw_text)} chars). Analyzing the first {max_chars} characters.")
                raw_text = raw_text[:max_chars]

            if st.button("🔍 Analyze File", type="primary"):
                with st.spinner("🧠 Analyzing the entire document for SOM patterns..."):
                    file_prompt = f"""You are an NLP expert analyzing a full document for Sleight of Mouth (SOM) patterns.

DOCUMENT:
{raw_text}

{ANALYSIS_SYSTEM_PROMPT}

TASK:
1. Read the entire document.
2. Extract ALL quotes or sentences that contain a clear SOM pattern.
3. For each, identify the pattern (English name), and explain it in the same language as the quote.
4. In general_reply (English), provide a brief overall summary: which patterns dominate, what it says about the author's communication style.
5. Plain sentences with no pattern → still include them (has_pattern=false) so we see full context.
"""
                    try:
                        resp = client.models.generate_content(
                            model='gemini-2.5-flash', contents=file_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=FullAnalysis)
                        )
                        analysis = json.loads(resp.text)

                        if analysis.get("general_reply"):
                            st.markdown(f"### 📊 Summary\n{analysis['general_reply']}")

                        st.divider()
                        st.markdown(f"**Found {len(analysis.get('items',[]))} lines analyzed:**")

                        pattern_counts = {}
                        for item in analysis.get("items", []):
                            if item.get("has_pattern"):
                                p = item.get("som_pattern", "Unknown")
                                pattern_counts[p] = pattern_counts.get(p, 0) + 1

                        if pattern_counts:
                            st.markdown("#### 🏆 Pattern Frequency")
                            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                            for pat, cnt in sorted_patterns:
                                st.markdown(f"- **{pat}**: {cnt} occurrence{'s' if cnt > 1 else ''}")
                            st.divider()

                        st.markdown("#### 📋 Full Breakdown")
                        for i_idx, item in enumerate(analysis.get("items", [])):
                            render_som_card(item, 9999, i_idx)

                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        else:
            st.error("Could not extract text from the file. Please try a different file.")
