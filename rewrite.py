import sys

def wrap_code(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('if "messages" not in st.session_state:'):
            start_idx = i
            break
            
    if start_idx == -1: return

    header = lines[:start_idx]
    body = lines[start_idx:]
    
    new_lines = header
    
    # Add navigation
    new_lines.extend([
        "with st.sidebar:\n",
        "    st.title('Navigation')\n",
        "    app_mode = st.radio('Choose Mode:', ['🗣️ Live Assistant', '🏋️ Training Simulator', '📁 File Analysis'])\n",
        "\n",
        "def render_live_assistant():\n"
    ])
    
    for line in body:
        if line == "\n":
            new_lines.append(line)
        else:
            new_lines.append("    " + line)
            
    new_lines.extend([
        "\n",
        "def render_training_simulator():\n",
        "    st.title('🏋️ Training Simulator')\n",
        "    st.info('Coming soon: Actively practice your Sleight of Mouth against generated objections.')\n",
        "\n",
        "def render_file_analysis():\n",
        "    st.title('📁 File Analysis')\n",
        "    st.info('Coming soon: Upload PDFs and Docs to extract frequently used patterns by the author.')\n",
        "\n",
        "if app_mode == '🗣️ Live Assistant':\n",
        "    render_live_assistant()\n",
        "elif app_mode == '🏋️ Training Simulator':\n",
        "    render_training_simulator()\n",
        "elif app_mode == '📁 File Analysis':\n",
        "    render_file_analysis()\n"
    ])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    wrap_code("E:/antigravity/som_agent_public/app.py")
