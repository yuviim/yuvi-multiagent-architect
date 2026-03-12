import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import time
import base64
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── PDF GENERATOR ────────────────────────────────────────
def convert_md_to_html(md: str) -> str:
    """Convert markdown to HTML manually."""
    html = md

    # Tables — convert | header | to <table>
    lines = html.split('\n')
    result = []
    in_table = False
    for i, line in enumerate(lines):
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # Check if next line is separator
            if i + 1 < len(lines) and re.match(r'[\|\-\s:]+', lines[i+1]):
                if not in_table:
                    result.append('<table><thead><tr>')
                    result.append(''.join(f'<th>{c}</th>' for c in cells))
                    result.append('</tr></thead><tbody>')
                    in_table = True
            elif re.match(r'^[\|\-\s:]+$', line.replace('|', '').strip()):
                pass  # skip separator line
            else:
                if not in_table:
                    result.append('<table><tbody>')
                    in_table = True
                result.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        else:
            if in_table:
                result.append('</tbody></table>')
                in_table = False
            result.append(line)
    if in_table:
        result.append('</tbody></table>')
    html = '\n'.join(result)

    # Headings
    html = re.sub(r'^#{4}\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^#{1}\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Code blocks
    html = re.sub(r'```[\w]*\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

    # Horizontal rules
    html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)

    # Bullet lists
    html = re.sub(r'^\s*[-•]\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group()}</ul>', html, flags=re.DOTALL)

    # Numbered lists
    html = re.sub(r'^\s*\d+\.\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Paragraphs — wrap plain text lines
    lines = html.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('<'):
            result.append(f'<p>{stripped}</p>')
        else:
            result.append(line)
    html = '\n'.join(result)

    return html


def markdown_to_pdf_html(md_text: str, title: str = "Solution Brief") -> str:
    """Wrap converted markdown in a clean printable HTML page."""
    body_html = convert_md_to_html(md_text)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600&family=Nunito:wght@700;800&display=swap');
  body {{ font-family: 'Lora', Georgia, serif; font-size: 13px; line-height: 1.8; color: #1e1b4b; max-width: 800px; margin: 0 auto; padding: 40px; }}
  h1, h2, h3, h4 {{ font-family: 'Nunito', sans-serif; color: #4f46e5; margin-top: 1.5rem; }}
  h1 {{ font-size: 2rem; border-bottom: 2px solid #c7d2fe; padding-bottom: 0.5rem; }}
  h2 {{ font-size: 1.4rem; }}
  h3 {{ font-size: 1.1rem; }}
  p  {{ margin: 0.6rem 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th {{ background: #eef2ff; color: #4f46e5; font-family: 'Nunito', sans-serif; padding: 8px 12px; text-align: left; border: 1px solid #c7d2fe; }}
  td {{ padding: 8px 12px; border: 1px solid #e2e5ef; }}
  tr:nth-child(even) td {{ background: #f8f9fc; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-family: monospace; }}
  pre  {{ background: #f8f9fc; padding: 1rem; border-radius: 8px; overflow-x: auto; }}
  ul   {{ padding-left: 1.5rem; }}
  li   {{ margin: 0.3rem 0; }}
  hr   {{ border: none; border-top: 1px solid #e2e5ef; margin: 1.5rem 0; }}
  strong {{ color: #312e81; }}
  .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e5ef; font-size: 0.75rem; color: #6b7280; text-align: center; }}
  @media print {{
    body {{ padding: 20px; }}
    .footer {{ position: fixed; bottom: 0; width: 100%; }}
  }}
</style>
</head>
<body>
{body_html}
<div class="footer">Generated by Yuvi's Agent · AI Solution Architect Platform</div>
</body>
</html>"""

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="Yuvi's Agent AI Architect",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLING ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lora:wght@400;500;600&display=swap');

:root {
    --bg:         #ffffff;
    --card:       #f8f9fc;
    --border:     #e2e5ef;
    --accent:     #4f46e5;
    --accent2:    #7c3aed;
    --green:      #059669;
    --amber:      #d97706;
    --text:       #1e1b4b;
    --muted:      #6b7280;
    --soft:       #eef2ff;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Lora', Georgia, serif !important;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 2rem 3rem !important; max-width: 1400px !important;}

h1, h2, h3 { font-family: 'Nunito', sans-serif !important; color: var(--text) !important; }

/* Header */
.maveric-header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2rem 0 2.5rem 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2.5rem;
}
.maveric-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; font-weight: 800;
    font-family: 'Nunito', sans-serif;
    color: white;
    box-shadow: 0 4px 15px rgba(79,70,229,0.25);
}
.maveric-title { font-family: 'Nunito', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--text); margin: 0; }
.maveric-subtitle { font-family: 'Lora', serif; font-size: 0.8rem; color: var(--muted); margin: 0; letter-spacing: 0.05em; }
.maveric-badge {
    margin-left: auto;
    background: var(--soft);
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.06em;
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    text-transform: uppercase;
}

/* Pipeline */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 1.5rem 0;
    padding: 1.5rem 2rem;
    background: var(--soft);
    border: 1px solid #c7d2fe;
    border-radius: 16px;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
    padding: 0.9rem 1.4rem;
    border-radius: 12px;
    border: 1.5px solid var(--border);
    background: white;
    min-width: 110px;
    transition: all 0.3s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.pipeline-node.active {
    border-color: var(--accent);
    background: var(--soft);
    box-shadow: 0 4px 16px rgba(79,70,229,0.15);
}
.pipeline-node.done {
    border-color: var(--green);
    background: #f0fdf4;
}
.pipeline-node.paused {
    border-color: var(--amber);
    background: #fffbeb;
    box-shadow: 0 4px 16px rgba(217,119,6,0.12);
}
.node-icon { font-size: 1.4rem; }
.node-label { font-family: 'Nunito', sans-serif; font-size: 0.7rem; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.node-status { font-size: 0.65rem; color: var(--muted); }
.node-status.active { color: var(--accent); font-weight: 700; }
.node-status.done { color: var(--green); font-weight: 700; }
.node-status.paused { color: var(--amber); font-weight: 700; }

.pipeline-arrow { font-size: 1.1rem; color: #c7d2fe; padding: 0 0.4rem; flex-shrink: 0; }
.pipeline-arrow.active { color: var(--accent); }

/* Cards */
.agent-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    color: var(--text);
}
.agent-card.highlight {
    border-color: #a5b4fc;
    box-shadow: 0 4px 16px rgba(79,70,229,0.08);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* Inputs */
.stTextArea textarea {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Lora', serif !important;
    font-size: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stTextInput input {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Lora', serif !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.25) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.35) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Status boxes */
.stSuccess { background: #f0fdf4 !important; border: 1px solid #bbf7d0 !important; border-radius: 10px !important; color: #14532d !important; }
.stInfo    { background: var(--soft) !important; border: 1px solid #c7d2fe !important; border-radius: 10px !important; color: #312e81 !important; }
.stWarning { background: #fffbeb !important; border: 1px solid #fde68a !important; border-radius: 10px !important; color: #78350f !important; }

/* Log terminal */
.log-terminal {
    background: #f8f9fc;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    max-height: 200px;
    overflow-y: auto;
    color: var(--text);
}
.log-supervisor { color: #4f46e5; font-weight: 600; }
.log-researcher { color: #0369a1; font-weight: 600; }
.log-architect  { color: #b45309; font-weight: 600; }
.log-writer     { color: #059669; font-weight: 600; }
.log-system     { color: #6b7280; }

/* Brief output */
.brief-container {
    background: white;
    border: 1.5px solid #a5b4fc;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(79,70,229,0.08);
    color: var(--text);
    font-family: 'Lora', serif;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--soft) !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="maveric-header">
    <div class="maveric-logo">M</div>
    <div>
        <div class="maveric-title">AI Solution Architect</div>
        <div class="maveric-subtitle">Yuvi's Agent · Multi-Agent Intelligence Platform</div>
    </div>
    <div class="maveric-badge">⬡ LangGraph · HITL · RAG-Ready</div>
</div>
""", unsafe_allow_html=True)

# ── AGENT PIPELINE DIAGRAM ───────────────────────────────
def render_pipeline(stage="idle"):
    stages = {
        "idle":       ("idle",       "idle",       "idle",       "idle"),
        "research":   ("active",     "idle",       "idle",       "idle"),
        "architect":  ("done",       "active",     "idle",       "idle"),
        "review":     ("done",       "done",       "paused",     "idle"),
        "writer":     ("done",       "done",       "done",       "active"),
        "complete":   ("done",       "done",       "done",       "done"),
    }
    s = stages.get(stage, stages["idle"])
    icons = ["🔬", "🏗️", "👤", "✍️"]
    labels = ["Researcher", "Architect", "You", "Writer"]
    statuses = {
        "idle":   ("○", ""),
        "active": ("◉", "active"),
        "done":   ("✓", "done"),
        "paused": ("⏸", "paused"),
    }

    nodes_html = ""
    for i, (st_val, icon, label) in enumerate(zip(s, icons, labels)):
        sym, cls = statuses[st_val]
        node_cls = f"pipeline-node {st_val}" if st_val != "idle" else "pipeline-node"
        status_cls = f"node-status {cls}" if cls else "node-status"
        nodes_html += f"""
        <div class="{node_cls}">
            <div class="node-icon">{icon}</div>
            <div class="node-label">{label}</div>
            <div class="{status_cls}">{sym}</div>
        </div>"""
        if i < len(s) - 1:
            arrow_cls = "pipeline-arrow active" if s[i] == "done" or s[i] == "active" else "pipeline-arrow"
            nodes_html += f'<div class="{arrow_cls}">──▶</div>'

    st.markdown(f'<div class="pipeline-container">{nodes_html}</div>', unsafe_allow_html=True)

# ── AGENT SETUP ──────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=3024)

class SolutionState(TypedDict):
    problem:        str
    research:       str
    architecture:   str
    human_feedback: str
    final_brief:    str
    next_agent:     str

def build_agent():
    model = get_model()
    parser = StrOutputParser()

    def supervisor_node(state: SolutionState) -> dict:
        if not state.get("research"):
            return {"next_agent": "researcher"}
        elif not state.get("architecture"):
            return {"next_agent": "architect"}
        elif not state.get("final_brief"):
            return {"next_agent": "writer"}
        else:
            return {"next_agent": "done"}

    def researcher_node(state: SolutionState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Research Analyst at Yuvi's Agent.
             Analyse the client problem and extract in bullet points (max 200 words):
             1. Root cause
             2. Key technical requirements
             3. Data landscape
             4. Compliance constraints
             Be concise."""),
            ("human", "Analyse this problem:\n{problem}")
        ])
        chain = prompt | model | parser
        research = chain.invoke({"problem": state["problem"]})
        return {"research": research}

    def architect_node(state: SolutionState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior AI Solution Architect at Yuvi's Agent.
             Design a solution in max 250 words covering:
             1. Architecture pattern and components
             2. Technology stack
             3. Security approach
             Be specific but concise."""),
            ("human", """Problem: {problem}
             Research: {research}
             Design the architecture.""")
        ])
        chain = prompt | model | parser
        architecture = chain.invoke({
            "problem":  state["problem"],
            "research": state["research"]
        })
        return {"architecture": architecture}

    def writer_node(state: SolutionState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Consultant at Yuvi's Agent.
             Synthesise the research and architecture into a client-facing brief.
             If human feedback is provided — incorporate it prominently.
             Structure:
             1. EXECUTIVE SUMMARY (3 sentences)
             2. RECOMMENDED SOLUTION
             3. KEY BENEFITS
             4. DELIVERY TIMELINE
             5. NEXT STEPS
             Write for C-suite. Clear, confident, no jargon."""),
            ("human", """Problem: {problem}
             Research: {research}
             Architecture: {architecture}
             Human feedback to incorporate: {human_feedback}
             Write the brief.""")
        ])
        chain = prompt | model | parser
        final_brief = chain.invoke({
            "problem":        state["problem"],
            "research":       state["research"],
            "architecture":   state["architecture"],
            "human_feedback": state.get("human_feedback", "No feedback — proceed as designed")
        })
        return {"final_brief": final_brief}

    def route_next(state: SolutionState) -> str:
        return state["next_agent"]

    graph = StateGraph(SolutionState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("architect",  architect_node)
    graph.add_node("writer",     writer_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_next, {
        "researcher": "researcher",
        "architect":  "architect",
        "writer":     "writer",
        "done":       END
    })
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("architect",  "supervisor")
    graph.add_edge("writer",     "supervisor")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory, interrupt_before=["writer"])

# ── SESSION STATE ─────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = "input"        # input | running | review | complete
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": f"session_{int(time.time())}"}}
if "logs" not in st.session_state:
    st.session_state.logs = []
if "architecture" not in st.session_state:
    st.session_state.architecture = ""
if "final_brief" not in st.session_state:
    st.session_state.final_brief = ""

def add_log(msg, cls="log-system"):
    st.session_state.logs.append(f'<span class="{cls}">{msg}</span>')

def render_logs():
    if st.session_state.logs:
        log_html = "<br>".join(st.session_state.logs)
        st.markdown(f'<div class="log-terminal">{log_html}</div>', unsafe_allow_html=True)

# ── LAYOUT — TOP TO BOTTOM FLOW ──────────────────────────

# ── ROW 1: PIPELINE DIAGRAM ──────────────────────────────
stage_map = {
    "input":    "idle",
    "research": "research",
    "architect":"architect",
    "review":   "review",
    "writing":  "writer",
    "complete": "complete",
}
render_pipeline(stage_map.get(st.session_state.stage, "idle"))

# ── ROW 2: PROBLEM INPUT ─────────────────────────────────
st.markdown("### 📋 Client Problem")
problem = st.text_area(
    "",
    value="""A large BPO company needs analysts to query 500K documents on SharePoint using natural language. Users should only see documents they are authorised to view. Answers needed in under 5 seconds with source citations.""",
    height=120,
    label_visibility="collapsed"
)

# ── ROW 3: CONTROLS ──────────────────────────────────────
if st.session_state.stage == "input":
    if st.button("⬡  Run Multi-Agent Analysis", use_container_width=True):
        st.session_state.stage = "research"
        st.session_state.logs = []
        st.session_state.config = {"configurable": {"thread_id": f"session_{int(time.time())}"}}
        st.rerun()

elif st.session_state.stage == "research":
    with st.spinner("Researcher analysing problem..."):
        add_log("▶ Agent pipeline started", "log-system")
        add_log("→ SUPERVISOR routing to RESEARCHER", "log-supervisor")
        add_log("◉ RESEARCHER analysing problem...", "log-researcher")
        st.session_state.agent.invoke({
            "problem":        problem,
            "research":       "",
            "architecture":   "",
            "human_feedback": "",
            "final_brief":    "",
            "next_agent":     ""
        }, config=st.session_state.config)
        add_log("✓ RESEARCHER complete", "log-researcher")
        add_log("→ SUPERVISOR routing to ARCHITECT", "log-supervisor")
        add_log("◉ ARCHITECT designing solution...", "log-architect")
        st.session_state.stage = "architect"
        st.rerun()

elif st.session_state.stage == "architect":
    current = st.session_state.agent.get_state(st.session_state.config)
    st.session_state.architecture = current.values.get("architecture", "")
    add_log("✓ ARCHITECT complete", "log-architect")
    add_log("⏸ SUPERVISOR pausing for human review", "log-supervisor")
    st.session_state.stage = "review"
    st.rerun()

elif st.session_state.stage == "review":
    st.warning("⏸️  **Human Review Required** — Agent paused before Writer")
    feedback = st.text_input(
        "Your feedback for the Writer (or leave blank to approve):",
        placeholder="e.g. Add more emphasis on RBAC and zero-trust security..."
    )
    col_approve, col_reset = st.columns([3, 1])
    with col_approve:
        if st.button("▶  Resume Agent", use_container_width=True):
            st.session_state.agent.update_state(
                st.session_state.config,
                {"human_feedback": feedback if feedback else "Approved — proceed as designed"}
            )
            feedback_log = f'"{feedback}"' if feedback else 'Approved'
            add_log(f"✓ Human feedback: {feedback_log}", "log-system")
            add_log("→ SUPERVISOR routing to WRITER", "log-supervisor")
            add_log("◉ WRITER producing client brief...", "log-writer")
            st.session_state.stage = "writing"
            st.rerun()
    with col_reset:
        if st.button("↺  Reset", use_container_width=True):
            for key in ["stage", "agent", "config", "logs", "architecture", "final_brief"]:
                del st.session_state[key]
            st.rerun()

elif st.session_state.stage == "writing":
    with st.spinner("Writer crafting client brief..."):
        result = st.session_state.agent.invoke(None, config=st.session_state.config)
        st.session_state.final_brief = result["final_brief"]
        add_log("✓ WRITER complete", "log-writer")
        add_log("✓ Pipeline complete — brief ready", "log-system")
        st.session_state.stage = "complete"
        st.rerun()

elif st.session_state.stage == "complete":
    st.success("✅ Multi-agent pipeline complete")
    if st.button("↺  New Analysis", use_container_width=True):
        for key in ["stage", "agent", "config", "logs", "architecture", "final_brief"]:
            del st.session_state[key]
        st.rerun()

# ── ROW 4: AGENT LOG ─────────────────────────────────────
if st.session_state.logs:
    st.markdown("### 🖥️ Agent Log")
    render_logs()

st.markdown("---")

# ── ROW 5: OUTPUT — ARCHITECTURE THEN BRIEF ──────────────
if st.session_state.stage in ["review", "writing", "complete"] and st.session_state.architecture:
    st.markdown("### 🏗️ Architecture Design")
    active = st.session_state.stage == "review"
    card_cls = "agent-card highlight" if active else "agent-card"
    st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
    st.markdown(st.session_state.architecture)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.stage == "complete" and st.session_state.final_brief:
    st.markdown("---")
    st.markdown("### 📄 Client Brief")
    st.markdown('<div class="brief-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_brief)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="⬇  Download as Markdown",
            data=st.session_state.final_brief,
            file_name="yuvi_solution_brief.md",
            mime="text/markdown",
            use_container_width=True
        )
    with dl2:
        html_content = markdown_to_pdf_html(st.session_state.final_brief)
        st.download_button(
            label="⬇  Download as PDF",
            data=html_content,
            file_name="yuvi_solution_brief.html",
            mime="text/html",
            use_container_width=True
        )
    st.caption("💡 For PDF: download HTML → open in Chrome → Cmd+P → Save as PDF")