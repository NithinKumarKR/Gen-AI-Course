import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="MathMind AI", page_icon="🧮", layout="wide")

# -------------------------------------------------------
# GLOBAL CSS — Dark Luxury Theme
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-deep:      #0b0c10;
    --bg-card:      #13141a;
    --bg-glass:     rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.08);
    --gold:         #d4a853;
    --gold-light:   #f0c97a;
    --gold-dim:     rgba(212,168,83,0.15);
    --teal:         #4ecdc4;
    --red-accent:   #ff6b6b;
    --text-primary: #f0ece4;
    --text-muted:   #7a7a8a;
    --radius:       14px;
}

/* ── GLOBAL RESET ── */
html, body, .stApp {
    background-color: var(--bg-deep) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Subtle grid texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(212,168,83,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(212,168,83,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1100px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* ── HERO HEADER ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-wrap::after {
    content: '';
    display: block;
    width: 160px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 2rem auto 0;
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid rgba(212,168,83,0.3);
    border-radius: 50px;
    padding: 5px 18px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -1px;
    color: var(--text-primary);
    margin: 0 0 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 0.3px;
    max-width: 480px;
    margin: 0 auto;
}

/* ── CAPABILITY PILLS ── */
.pill-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 1.8rem 0 0;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12.5px;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 50px;
    border: 1px solid var(--border);
    background: var(--bg-glass);
    color: var(--text-muted);
    backdrop-filter: blur(8px);
}
.pill-math   { border-color: rgba(212,168,83,0.35); color: var(--gold); }
.pill-reason { border-color: rgba(78,205,196,0.35); color: var(--teal); }
.pill-wiki   { border-color: rgba(255,107,107,0.35); color: var(--red-accent); }

/* ── STAT CARDS ROW ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 2.5rem 0;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.3s;
}
.stat-card:hover { border-color: rgba(212,168,83,0.3); }
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    font-size: 11.5px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── DIVIDER ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── CHAT MESSAGES ── */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.4rem !important;
    margin-bottom: 10px !important;
}
.stChatMessage[data-testid="chat-message-assistant"] {
    border-left: 3px solid var(--gold) !important;
}
.stChatMessage[data-testid="chat-message-user"] {
    border-left: 3px solid var(--teal) !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold) !important;
}
[data-testid="stSidebarContent"] {
    padding: 2rem 1.4rem !important;
}

/* Sidebar section label */
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin: 1.6rem 0 0.5rem;
    opacity: 0.8;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: var(--gold) !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold), #b8892e) !important;
    color: #0b0c10 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── TEXT INPUTS ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
}

/* ── INFO / ALERT BOX ── */
[data-testid="stAlert"] {
    background: var(--gold-dim) !important;
    border: 1px solid rgba(212,168,83,0.3) !important;
    border-radius: var(--radius) !important;
    color: var(--gold-light) !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* ── TOOL TAG ── */
.tool-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    padding: 3px 12px;
    border-radius: 50px;
    margin-bottom: 8px;
    font-weight: 500;
}
.tool-math     { background: rgba(212,168,83,0.15); color: var(--gold); border: 1px solid rgba(212,168,83,0.3); }
.tool-reason   { background: rgba(78,205,196,0.12); color: var(--teal); border: 1px solid rgba(78,205,196,0.3); }
.tool-wiki     { background: rgba(255,107,107,0.12); color: var(--red-accent); border: 1px solid rgba(255,107,107,0.3); }

/* ── EXAMPLE QUESTIONS CARD ── */
.examples-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin: 1.5rem 0 2rem;
}
.examples-card h4 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: var(--gold);
    margin: 0 0 1rem;
    font-weight: 700;
}
.ex-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
    color: var(--text-muted);
    line-height: 1.5;
}
.ex-item:last-child { border-bottom: none; }
.ex-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}
.dot-gold { background: var(--gold); }
.dot-teal { background: var(--teal); }
.dot-red  { background: var(--red-accent); }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(212,168,83,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ── FADE-IN ANIMATION ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-wrap  { animation: fadeUp 0.6s ease both; }
.stat-row   { animation: fadeUp 0.6s 0.15s ease both; }
.examples-card { animation: fadeUp 0.6s 0.25s ease both; }

/* ── FOOTER ── */
.page-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
.page-footer span { color: var(--gold); }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HERO HEADER
# -------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦ Powered by Groq LLM</div>
    <h1 class="hero-title">Math<span>Mind</span> AI</h1>
    <p class="hero-subtitle">Your intelligent companion for mathematics, logical reasoning, and knowledge discovery.</p>
    <div class="pill-row">
        <span class="pill pill-math">🧮 Mathematics</span>
        <span class="pill pill-reason">🧠 Reasoning</span>
        <span class="pill pill-wiki">📖 Knowledge</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# STAT CARDS
# -------------------------------------------------------
st.markdown("""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-num">3</div>
        <div class="stat-label">Specialist Engines</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">∞</div>
        <div class="stat-label">Questions Answered</div>
    </div>
    <div class="stat-card">
        <div class="stat-num">AI</div>
        <div class="stat-label">Smart Routing</div>
    </div>
</div>
<hr class="fancy-divider">
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom:1.2rem; border-bottom:1px solid rgba(255,255,255,0.08);">
        <div style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#d4a853;">MathMind</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:2px; color:#7a7a8a; text-transform:uppercase; margin-top:2px;">Configuration Panel</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">🔑 Authentication</div>', unsafe_allow_html=True)
    GROQ_API_KEY = st.text_input("GROQ API Key", type="password", placeholder="gsk_...")

    st.markdown('<div class="sidebar-section">🤖 Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Select Model", [
        "Gemma2-9b-It",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">⚙️ Parameters</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05,
                            help="Higher = more creative answers")
    max_token = st.slider("Max Tokens", 100, 4096, 1024,
                          help="Maximum response length")

    st.markdown('<div class="sidebar-section">📊 Session</div>', unsafe_allow_html=True)
    msg_count = len(st.session_state.get("messages", [])) - 1
    st.markdown(f"""
    <div style="background:rgba(212,168,83,0.08); border:1px solid rgba(212,168,83,0.2);
                border-radius:10px; padding:0.9rem 1rem; font-size:13px; color:#d4a853;">
        💬 Messages this session: <strong>{max(0, msg_count)}</strong>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑 Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything 🧮"}
        ]
        st.rerun()

    st.markdown("""
    <div style="margin-top:2rem; padding:1rem; background:rgba(255,255,255,0.03);
                border-radius:10px; border:1px solid rgba(255,255,255,0.06); font-size:12px; color:#7a7a8a; line-height:1.6;">
        <strong style="color:#d4a853;">How routing works</strong><br>
        The AI classifier reads your question and picks the best engine automatically:<br><br>
        🧮 <em style="color:#d4a853;">Math</em> — calculations & equations<br>
        🧠 <em style="color:#4ecdc4;">Reasoning</em> — logic & analysis<br>
        📖 <em style="color:#ff6b6b;">Wiki</em> — facts & knowledge
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# GUARD: Stop if no API key
# -------------------------------------------------------
if not GROQ_API_KEY:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; background:rgba(212,168,83,0.05);
                border:1px dashed rgba(212,168,83,0.3); border-radius:16px; margin: 1rem 0 2rem;">
        <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔑</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.3rem; color:#d4a853; margin-bottom:0.5rem;">API Key Required</div>
        <div style="color:#7a7a8a; font-size:14px; max-width:340px; margin:0 auto; line-height:1.6;">
            Enter your <strong style="color:#f0c97a;">GROQ API Key</strong> in the sidebar to unlock the assistant.
            Get a free key at <em>console.groq.com</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example questions preview
    st.markdown("""
    <div class="examples-card">
        <h4>✦ What you can ask</h4>
        <div class="ex-item"><div class="ex-dot dot-gold"></div><div>Solve: ∫(x² + 3x) dx from 0 to 5</div></div>
        <div class="ex-item"><div class="ex-dot dot-gold"></div><div>What is the probability that two people share a birthday in a group of 30?</div></div>
        <div class="ex-item"><div class="ex-dot dot-teal"></div><div>If all roses are flowers and some flowers fade quickly, do all roses fade?</div></div>
        <div class="ex-item"><div class="ex-dot dot-teal"></div><div>Alice is taller than Bob who is taller than Carol. Who is shortest?</div></div>
        <div class="ex-item"><div class="ex-dot dot-red"></div><div>What is the theory of general relativity?</div></div>
        <div class="ex-item"><div class="ex-dot dot-red"></div><div>Who discovered the structure of DNA and when?</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
model = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_token,
    streaming=False
)

# -------------------------------------------------------
# TOOL CHAINS
# -------------------------------------------------------

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)

def wiki_tool(question: str) -> str:
    return wiki_wrapper.run(question)

math_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a math expert. Solve the problem step by step and give the final answer clearly."),
        ("human", "{question}")
    ])
    | model
    | StrOutputParser()
)

def math_tool(question: str) -> str:
    return math_chain.invoke({"question": question})

reasoning_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a logical reasoning expert. Think step by step and explain your reasoning clearly."),
        ("human", "{question}")
    ])
    | model
    | StrOutputParser()
)

def reasoning_tool(question: str) -> str:
    return reasoning_chain.invoke({"question": question})

# -------------------------------------------------------
# ROUTER CHAIN
# -------------------------------------------------------
router_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a question classifier. Given a user question, decide which tool to use.

Reply with ONLY one of these exact words (no explanation):
- "math"      → for arithmetic, algebra, calculus, geometry, statistics, or any numerical computation
- "reasoning" → for logic puzzles, word problems requiring step-by-step thinking, or analytical questions
- "wiki"      → for factual / general knowledge questions about people, places, events, science concepts, history

Question: {question}
Tool:"""),
        ("human", "{question}")
    ])
    | model
    | StrOutputParser()
)

TOOL_MAP = {
    "math": math_tool,
    "reasoning": reasoning_tool,
    "wiki": wiki_tool,
}

TOOL_LABEL = {
    "math":      ("🧮", "Calculator",  "tool-math"),
    "reasoning": ("🧠", "Reasoning",   "tool-reason"),
    "wiki":      ("📖", "Wikipedia",   "tool-wiki"),
}

# -------------------------------------------------------
# GENERATE RESPONSE
# -------------------------------------------------------
def generate_response(user_question: str) -> tuple[str, str]:
    """Returns (formatted_answer, tool_key)"""
    try:
        tool_key = router_chain.invoke({"question": user_question}).strip().lower()

        if tool_key not in TOOL_MAP:
            if any(k in tool_key for k in TOOL_MAP):
                tool_key = next(k for k in TOOL_MAP if k in tool_key)
            else:
                tool_key = "reasoning"

        selected_tool = TOOL_MAP[tool_key]
        result = selected_tool(user_question)
        return result, tool_key

    except Exception as e:
        return f"❌ Error: {str(e)}", "reasoning"

# -------------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm **MathMind**, your AI companion for mathematics, reasoning, and knowledge.\n\nAsk me anything — I'll automatically route your question to the best specialist engine.", "tool": None}
    ]

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool"):
            icon, label, css_class = TOOL_LABEL.get(msg["tool"], ("🧠", "AI", "tool-reason"))
            st.markdown(f'<span class="tool-tag {css_class}">{icon} {label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

# -------------------------------------------------------
# CHAT INPUT
# -------------------------------------------------------
if user_input := st.chat_input("Ask a math, logic, or knowledge question…"):
    st.session_state["messages"].append({"role": "user", "content": user_input, "tool": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, tool_key = generate_response(user_input)
            icon, label, css_class = TOOL_LABEL.get(tool_key, ("🧠", "AI", "tool-reason"))
            st.markdown(f'<span class="tool-tag {css_class}">{icon} {label}</span>', unsafe_allow_html=True)
            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer, "tool": tool_key})

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("""
<div class="page-footer">
    Built with <span>♥</span> using Streamlit · LangChain · Groq &nbsp;·&nbsp; MathMind AI
</div>
""", unsafe_allow_html=True)