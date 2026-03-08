# ============================================================
#  L E X I S  A I  —  Intelligent Document Oracle
#  Ultra Premium UI Edition
# ============================================================

import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_astradb import AstraDBVectorStore


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Lexis AI — Intelligent Document Oracle",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MASTER CSS — Electric Obsidian Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════
   ROOT VARIABLES
══════════════════════════════════════════ */
:root {
    --obsidian:       #080a0f;
    --deep:           #0c0e15;
    --surface:        #111420;
    --card:           #161925;
    --card-hover:     #1c2030;
    --raised:         #1f2335;

    --cyan:           #00d4ff;
    --cyan-dim:       #00a8cc;
    --cyan-glow:      rgba(0, 212, 255, 0.18);
    --cyan-border:    rgba(0, 212, 255, 0.22);
    --cyan-border-hv: rgba(0, 212, 255, 0.55);

    --teal:           #00ffcc;
    --coral:          #ff6b6b;
    --amber:          #ffb347;

    --text:           #e8eaf0;
    --text-dim:       #8b92a8;
    --text-ghost:     #4a5168;

    --success-bg:     rgba(0, 255, 136, 0.08);
    --success-border: rgba(0, 255, 136, 0.3);
    --success-text:   #00ff88;

    --r:  10px;
    --rl: 16px;
    --rxl: 24px;
}

/* ══════════════════════════════════════════
   GLOBAL BASE
══════════════════════════════════════════ */
html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: var(--obsidian) !important;
    color: var(--text) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

#MainMenu, footer, header, .stDeployButton { display: none !important; visibility: hidden !important; }

.main .block-container {
    padding: 1.8rem 2.5rem 5rem !important;
    max-width: 1160px !important;
    position: relative;
    z-index: 1;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--cyan-border) !important;
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.1rem !important;
}

/* Sidebar section headers */
.sb-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--cyan);
    opacity: 0.7;
    padding: 0.5rem 0 0.4rem;
    margin: 0.8rem 0 0.5rem;
    border-bottom: 1px solid var(--cyan-border);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Sidebar doc info card */
.doc-card {
    background: var(--card);
    border: 1px solid var(--cyan-border);
    border-radius: var(--r);
    padding: 0.85rem 1rem;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}
.doc-card .doc-name {
    color: var(--cyan);
    font-weight: 600;
    font-size: 0.82rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 0.55rem;
}
.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid var(--cyan-border);
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    font-size: 0.73rem;
    color: var(--text-dim);
    margin-right: 0.4rem;
}

/* ══════════════════════════════════════════
   MASTHEAD
══════════════════════════════════════════ */
.masthead {
    position: relative;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: var(--rxl);
    overflow: hidden;
    background: linear-gradient(135deg, #0c0e18 0%, #080a10 50%, #0d1018 100%);
    border: 1px solid var(--cyan-border);
}
.masthead::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 65%);
    pointer-events: none;
}
.masthead::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,255,204,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.masthead-inner {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    z-index: 2;
}
.masthead-logo {
    width: 60px; height: 60px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,255,204,0.08));
    border: 1px solid var(--cyan-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 0 30px rgba(0,212,255,0.2);
    flex-shrink: 0;
}
.masthead-copy { flex: 1; }
.masthead-copy h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    margin: 0 !important;
    line-height: 1 !important;
    letter-spacing: -0.01em;
}
.masthead-copy h1 span {
    background: linear-gradient(90deg, var(--cyan), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.masthead-copy p {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    margin: 0.45rem 0 0 !important;
    font-weight: 300;
    letter-spacing: 0.01em;
}
.masthead-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
    flex-shrink: 0;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.status-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
.pill-ready  { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
.pill-ready .dot { background: var(--success-text); box-shadow: 0 0 6px var(--success-text); }
.pill-idle   { background: rgba(74,81,104,0.2); color: var(--text-ghost); border: 1px solid rgba(74,81,104,0.3); }
.pill-idle .dot { background: var(--text-ghost); animation: none; }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

.model-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    color: var(--text-ghost);
    background: var(--card);
    border: 1px solid rgba(74,81,104,0.3);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
}

/* ══════════════════════════════════════════
   QUICK STATS BAR
══════════════════════════════════════════ */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-tile {
    flex: 1;
    padding: 1rem 1.2rem;
    background: var(--card);
    border: 1px solid var(--cyan-border);
    border-radius: var(--rl);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.stat-tile:hover {
    transform: translateY(-3px);
    border-color: var(--cyan-border-hv);
    box-shadow: 0 8px 32px rgba(0,212,255,0.1);
}
.stat-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--teal));
    opacity: 0.6;
}
.stat-icon  { font-size: 1.3rem; margin-bottom: 0.5rem; }
.stat-num   {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--cyan);
    line-height: 1;
}
.stat-lbl   {
    font-size: 0.71rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-top: 0.3rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: var(--rl) var(--rl) 0 0 !important;
    border: 1px solid var(--cyan-border) !important;
    border-bottom: none !important;
    padding: 0.5rem !important;
    gap: 0.25rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--cyan) !important;
    background: var(--cyan-glow) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--cyan-glow) !important;
    color: var(--cyan) !important;
    box-shadow: inset 0 0 0 1px var(--cyan-border) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--card) !important;
    border: 1px solid var(--cyan-border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--rl) var(--rl) !important;
    padding: 1.5rem !important;
}

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, var(--teal) 100%) !important;
    color: #050710 !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 36px rgba(0,212,255,0.4) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }
.stButton > button:disabled {
    background: rgba(74,81,104,0.2) !important;
    color: var(--text-ghost) !important;
    box-shadow: none !important;
    transform: none !important;
    filter: none !important;
}
.ghost-btn > button {
    background: transparent !important;
    color: var(--text-dim) !important;
    border: 1px solid rgba(74,81,104,0.4) !important;
    box-shadow: none !important;
}
.ghost-btn > button:hover {
    border-color: var(--cyan-border-hv) !important;
    color: var(--cyan) !important;
    transform: none !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.1) !important;
    filter: none !important;
}

/* ══════════════════════════════════════════
   INPUTS
══════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid rgba(74,81,104,0.4) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.93rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    caret-color: var(--cyan);
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-ghost) !important; }

/* ══════════════════════════════════════════
   SELECTBOX / SLIDERS
══════════════════════════════════════════ */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid rgba(74,81,104,0.4) !important;
    border-radius: var(--r) !important;
    color: var(--text) !important;
}
.stSlider > div > div > div > div { background: var(--cyan) !important; }

/* ══════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--cyan-border-hv) !important;
    border-radius: var(--rxl) !important;
    background: var(--surface) !important;
    transition: all 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    background: var(--cyan-glow) !important;
    box-shadow: 0 0 40px rgba(0,212,255,0.08) !important;
}

/* ══════════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════════ */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    max-height: 500px;
    overflow-y: auto;
    padding: 0.75rem 0.25rem 0.25rem;
    scrollbar-width: thin;
    scrollbar-color: var(--cyan-border) transparent;
}
.chat-wrap::-webkit-scrollbar { width: 4px; }
.chat-wrap::-webkit-scrollbar-thumb { background: var(--cyan-border); border-radius: 10px; }

.chat-row { display: flex; gap: 0.85rem; animation: msgIn 0.35s cubic-bezier(.22,1,.36,1) forwards; }
.chat-row.from-user { flex-direction: row-reverse; }

@keyframes msgIn {
    from { opacity: 0; transform: translateY(14px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}

.av {
    width: 38px; height: 38px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.av-user { background: linear-gradient(135deg, #1a3a6b, #2563eb); border: 1px solid rgba(37,99,235,0.4); }
.av-bot  {
    background: linear-gradient(135deg, #051a1f, #063a45);
    border: 1px solid var(--cyan-border);
    box-shadow: 0 0 16px rgba(0,212,255,0.15);
}

.bubble {
    max-width: 72%;
    padding: 0.9rem 1.15rem;
    border-radius: var(--rl);
    font-size: 0.9rem;
    line-height: 1.7;
}
.bubble-user {
    background: linear-gradient(135deg, #142448, #1a2f5c);
    border: 1px solid rgba(37,99,235,0.25);
    border-bottom-right-radius: 4px;
    color: #c7d9ff;
}
.bubble-bot {
    background: var(--raised);
    border: 1px solid var(--cyan-border);
    border-bottom-left-radius: 4px;
    color: var(--text);
}
.bubble-footer {
    font-size: 0.68rem;
    color: var(--text-ghost);
    margin-top: 0.45rem;
    font-family: 'IBM Plex Mono', monospace;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.from-user .bubble-footer { justify-content: flex-end; }

/* speed badge */
.speed-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid var(--cyan-border);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.65rem;
    color: var(--cyan-dim);
    font-family: 'IBM Plex Mono', monospace;
}

/* word count badge */
.wc-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: rgba(74,81,104,0.15);
    border: 1px solid rgba(74,81,104,0.3);
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
    font-size: 0.65rem;
    color: var(--text-ghost);
    font-family: 'IBM Plex Mono', monospace;
}

/* ══════════════════════════════════════════
   PROCESS STEPS
══════════════════════════════════════════ */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.7rem 1rem;
    border-radius: var(--r);
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
    font-weight: 500;
    border: 1px solid transparent;
    transition: all 0.2s;
}
.pipe-done   { background: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
.pipe-active { background: var(--cyan-glow);  color: var(--cyan);         border-color: var(--cyan-border); }
.pipe-idle   { background: rgba(74,81,104,0.1); color: var(--text-ghost); border-color: rgba(74,81,104,0.2); }
.pipe-num {
    width: 26px; height: 26px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    flex-shrink: 0;
}
.pipe-done   .pipe-num { background: var(--success-bg);  border: 1px solid var(--success-border); color: var(--success-text); }
.pipe-active .pipe-num { background: var(--cyan-glow);   border: 1px solid var(--cyan-border); color: var(--cyan); }
.pipe-idle   .pipe-num { background: rgba(74,81,104,0.15); border: 1px solid rgba(74,81,104,0.3); color: var(--text-ghost); }
.pipe-label { flex: 1; }
.pipe-check { font-size: 0.85rem; margin-left: auto; }

/* ══════════════════════════════════════════
   SOURCE CHUNKS
══════════════════════════════════════════ */
.chunk-card {
    background: var(--surface);
    border: 1px solid var(--cyan-border);
    border-left: 3px solid var(--cyan);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.82rem;
    color: var(--text-dim);
    line-height: 1.65;
    transition: border-color 0.2s;
}
.chunk-card:hover { border-color: var(--cyan-border-hv); }
.chunk-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--cyan);
    display: block;
    margin-bottom: 0.45rem;
    opacity: 0.8;
}

/* ══════════════════════════════════════════
   EMPTY STATES
══════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 3.5rem 1.5rem;
}
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-title { font-size: 1rem; font-weight: 600; color: var(--text-dim); margin-bottom: 0.4rem; }
.empty-sub   { font-size: 0.83rem; color: var(--text-ghost); }
.empty-sub strong { color: var(--cyan); font-weight: 500; }

/* ══════════════════════════════════════════
   UPLOAD PREVIEW CARD
══════════════════════════════════════════ */
.file-preview {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    background: var(--surface);
    border: 1px solid var(--cyan-border);
    border-radius: var(--rl);
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
    transition: border-color 0.2s;
}
.file-preview:hover { border-color: var(--cyan-border-hv); }
.file-icon {
    width: 48px; height: 48px;
    background: var(--cyan-glow);
    border: 1px solid var(--cyan-border);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}
.file-name  { font-weight: 600; font-size: 0.93rem; color: var(--text); margin-bottom: 0.25rem; }
.file-meta  { font-size: 0.76rem; color: var(--text-dim); font-family: 'IBM Plex Mono', monospace; }

/* ══════════════════════════════════════════
   SECTION DIVIDER LABELS
══════════════════════════════════════════ */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--cyan-border), transparent);
}

/* ══════════════════════════════════════════
   SUGGESTION CHIPS
══════════════════════════════════════════ */
.stButton.sug > button {
    background: var(--surface) !important;
    color: var(--text-dim) !important;
    border: 1px solid rgba(74,81,104,0.35) !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
.stButton.sug > button:hover {
    border-color: var(--cyan-border-hv) !important;
    color: var(--cyan) !important;
    background: var(--cyan-glow) !important;
    transform: none !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.08) !important;
}

/* ══════════════════════════════════════════
   HISTORY PAIRS
══════════════════════════════════════════ */
.hist-q {
    background: var(--surface);
    border: 1px solid rgba(74,81,104,0.3);
    border-radius: var(--r) var(--r) 0 0;
    padding: 0.85rem 1rem;
}
.hist-a {
    background: var(--card);
    border: 1px solid var(--cyan-border);
    border-top: none;
    border-radius: 0 0 var(--r) var(--r);
    padding: 0.85rem 1rem;
    margin-bottom: 1rem;
}
.hist-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.4rem;
}

/* ══════════════════════════════════════════
   ALERTS / METRICS / MISC
══════════════════════════════════════════ */
.stAlert {
    border-radius: var(--r) !important;
    background: var(--card) !important;
    border: 1px solid var(--cyan-border) !important;
}
hr { border-color: rgba(74,81,104,0.25) !important; margin: 1.2rem 0 !important; }
label { color: var(--text-dim) !important; font-size: 0.82rem !important; }
.stCaption { color: var(--text-ghost) !important; font-size: 0.77rem !important; font-family: 'IBM Plex Mono', monospace !important; }
.stSpinner > div { border-top-color: var(--cyan) !important; }

[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: var(--r) !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label { color: var(--text-ghost) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'Playfair Display', serif !important; font-size: 1.6rem !important; }

[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: var(--r) !important;
}
[data-testid="stExpander"] summary { color: var(--text-dim) !important; font-size: 0.84rem !important; }

* { scrollbar-width: thin; scrollbar-color: var(--cyan-border) transparent; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "chat_history":       [],
    "vectors":            None,
    "doc_stats":          {"pages": 0, "chunks": 0, "filename": None, "size_kb": 0},
    "total_queries":      0,
    "response_times":     [],
    "total_words_read":   0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-head">⚡ Credentials</div>', unsafe_allow_html=True)
    LANGCHAIN_API_KEY = st.text_input("LangChain API Key", type="password", placeholder="lsv2_…")
    GROQ_API_KEY      = st.text_input("Groq API Key",      type="password", placeholder="gsk_…")
    HF_TOKEN          = st.text_input("HuggingFace Token", type="password", placeholder="hf_…")
    astra_token       = st.text_input("AstraDB Token",     type="password", placeholder="AstraCS:…")
    astra_endpoint    = st.text_input("AstraDB Endpoint",  type="password",
                                       placeholder="https://<db-id>.apps.astra.datastax.com")

    st.markdown('<div class="sb-head">🎛️ Model Config</div>', unsafe_allow_html=True)
    model_name  = st.selectbox("LLM Engine", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
                                help="70b = more intelligent · 8b = faster")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1,
                             help="Low = precise & factual · High = creative & expansive")
    max_tokens  = st.slider("Max Response Length", 64, 2048, 768, 64)
    top_k       = st.slider("Retrieved Chunks (Top-K)", 2, 8, 4,
                             help="How many document segments to retrieve per question")

    st.markdown('<div class="sb-head">📊 Session Intelligence</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    avg_rt = (sum(st.session_state.response_times) / len(st.session_state.response_times)
              if st.session_state.response_times else 0)
    c2.metric("Avg Speed", f"{avg_rt:.1f}s")
    st.metric("Words Analysed", f"{st.session_state.total_words_read:,}")

    st.markdown('<div class="sb-head">🗑️ Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.chat_history     = []
        st.session_state.total_queries    = 0
        st.session_state.response_times   = []
        st.session_state.total_words_read = 0
        st.rerun()
    if st.button("🔄 Reset Everything", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v if not isinstance(v, dict) else v.copy()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Loaded document card
    if st.session_state.doc_stats["filename"]:
        st.markdown('<div class="sb-head">📄 Active Document</div>', unsafe_allow_html=True)
        ds = st.session_state.doc_stats
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-name">📄 {ds['filename']}</div>
            <div>
                <span class="doc-pill">📑 {ds['pages']} pages</span>
                <span class="doc-pill">🧩 {ds['chunks']} chunks</span>
            </div>
            <div style="margin-top:0.5rem;">
                <span class="doc-pill">💾 {ds['size_kb']:.0f} KB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ENV VARS
# ============================================================
def set_env_vars():
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "Lexis AI"
    os.environ["GROQ_API_KEY"]         = GROQ_API_KEY or ""
    os.environ["HF_TOKEN"]             = HF_TOKEN or ""

set_env_vars()


# ============================================================
# MODEL & EMBEDDINGS — api_key in signature busts cache on change
# ============================================================
@st.cache_resource
def get_llm(model, temp, max_tok, api_key):
    return ChatGroq(model_name=model, temperature=temp, max_tokens=max_tok, api_key=api_key)

@st.cache_resource
def get_embeddings(hf_token):
    return HuggingFaceEndpointEmbeddings(
        task="feature-extraction",
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
    )

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are Lexis, a precise and intelligent document analyst.
Answer the question using ONLY the provided context.
Structure your answer clearly. Be thorough but concise.
If the answer isn't in the context, say so honestly.

<context>
{context}
</context>

Question: {input}

Answer:""")


# ============================================================
# HELPERS
# ============================================================
def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    pages = PyPDFLoader(tmp_path).load_and_split()
    return pages

def build_vector_store(pages, token, endpoint):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(pages)
    vs = AstraDBVectorStore(
        embedding=get_embeddings(HF_TOKEN),
        collection_name="lexis_ai_store",
        token=token,
        api_endpoint=endpoint,
    )
    vs.add_documents(chunks)
    return vs, len(chunks)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag(question, retriever, llm):
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    t0     = time.time()
    answer = chain.invoke(question)
    return answer, round(time.time() - t0, 2)

def word_count(text):
    return len(text.split())

def reading_time(text):
    wc = word_count(text)
    secs = int((wc / 238) * 60)
    return f"{secs}s read" if secs < 60 else f"{secs//60}m {secs%60}s read"


# ============================================================
# ─── MASTHEAD ───────────────────────────────────────────────
# ============================================================
db_ready = st.session_state.vectors is not None
pill_cls = "pill-ready" if db_ready else "pill-idle"
pill_txt = "Oracle Active" if db_ready else "Awaiting Document"
dot_html = '<span class="dot"></span>'

st.markdown(f"""
<div class="masthead">
  <div class="masthead-inner">
    <div class="masthead-logo">⚡</div>
    <div class="masthead-copy">
      <h1>Lexis <span>AI</span></h1>
      <p>Intelligent Document Oracle &mdash; Extract knowledge from any PDF instantly</p>
    </div>
    <div class="masthead-meta">
      <div class="status-pill {pill_cls}">{dot_html}{pill_txt}</div>
      <div class="model-tag">▸ {model_name}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ────── Quick Stats Bar ─────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
  <div class="stat-tile">
    <div class="stat-icon">📑</div>
    <div class="stat-num">{st.session_state.doc_stats['pages']}</div>
    <div class="stat-lbl">Pages Indexed</div>
  </div>
  <div class="stat-tile">
    <div class="stat-icon">🧩</div>
    <div class="stat-num">{st.session_state.doc_stats['chunks']}</div>
    <div class="stat-lbl">Knowledge Chunks</div>
  </div>
  <div class="stat-tile">
    <div class="stat-icon">💬</div>
    <div class="stat-num">{st.session_state.total_queries}</div>
    <div class="stat-lbl">Queries Answered</div>
  </div>
  <div class="stat-tile">
    <div class="stat-icon">⚡</div>
    <div class="stat-num">{avg_rt:.1f}s</div>
    <div class="stat-lbl">Avg Response</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_ingest, tab_oracle, tab_logbook, tab_export = st.tabs([
    "⬆️  Ingest Document",
    "🔮  Oracle Chat",
    "📜  Session Logbook",
    "📤  Export",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — INGEST DOCUMENT
# ════════════════════════════════════════════════════════════
with tab_ingest:
    astra_ready = bool(astra_token and astra_endpoint)
    if not astra_ready:
        st.warning("⚠️ Enter your **AstraDB Token** and **Endpoint** in the sidebar to continue.")

    uploaded_file = st.file_uploader(
        "⬆️  Drop your PDF — or click to browse",
        type="pdf",
        label_visibility="visible",
    )

    if uploaded_file:
        fsize_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="file-preview">
          <div class="file-icon">📄</div>
          <div>
            <div class="file-name">{uploaded_file.name}</div>
            <div class="file-meta">PDF Document &nbsp;·&nbsp; {fsize_kb:.1f} KB &nbsp;·&nbsp; Ready to ingest</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Ingestion Pipeline</div>', unsafe_allow_html=True)

    steps = [
        (1, "📤", "Document Upload",      "done"   if uploaded_file else "idle"),
        (2, "✂️", "Semantic Chunking",    "active" if (uploaded_file and not db_ready) else ("done" if db_ready else "idle")),
        (3, "🔢", "Vector Embedding",     "active" if (uploaded_file and not db_ready) else ("done" if db_ready else "idle")),
        (4, "🗄️", "AstraDB Persistence",  "done"   if db_ready else "idle"),
        (5, "✅", "Oracle Ready",         "done"   if db_ready else "idle"),
    ]
    for num, icon, label, state in steps:
        check = "✅" if state == "done" else ("⚡" if state == "active" else "○")
        st.markdown(f"""
        <div class="pipe-step pipe-{state}">
          <div class="pipe-num">{num}</div>
          <span style="font-size:1rem;">{icon}</span>
          <span class="pipe-label">{label}</span>
          <span class="pipe-check">{check}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡  Begin Ingestion", disabled=(uploaded_file is None or not astra_ready),
                 use_container_width=True):
        pb = st.progress(0, text="Initialising…")
        try:
            pb.progress(15, text="📖  Parsing PDF pages…")
            pages = load_pdf(uploaded_file)

            pb.progress(40, text="✂️  Chunking into semantic segments…")
            time.sleep(0.2)

            pb.progress(65, text="🔢  Generating vector embeddings…")
            vs, n_chunks = build_vector_store(pages, astra_token, astra_endpoint)

            pb.progress(90, text="🗄️  Persisting to AstraDB…")
            time.sleep(0.15)

            pb.progress(100, text="✅  Ingestion complete!")
            st.session_state.vectors   = vs
            fsize_kb = len(uploaded_file.getvalue()) / 1024
            st.session_state.doc_stats = {
                "pages": len(pages), "chunks": n_chunks,
                "filename": uploaded_file.name, "size_kb": fsize_kb,
            }
            st.success(f"✅  **{uploaded_file.name}** is now in the Oracle's knowledge base.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages",      len(pages))
            c2.metric("Chunks",     n_chunks)
            c3.metric("Chunk Size", "1 000 chars")
            c4.metric("Overlap",    "200 chars")
            st.rerun()

        except Exception as e:
            st.error(f"❌  Ingestion failed: {e}")


# ════════════════════════════════════════════════════════════
# TAB 2 — ORACLE CHAT
# ════════════════════════════════════════════════════════════
with tab_oracle:
    if not db_ready:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🔮</div>
          <div class="empty-title">The Oracle awaits a document</div>
          <div class="empty-sub">Go to <strong>⬆️ Ingest Document</strong> to load your PDF first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render conversation ──────────────────────────────
        if st.session_state.chat_history:
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                role = msg["role"]
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-row from-user">
                      <div class="av av-user">👤</div>
                      <div>
                        <div class="bubble bubble-user">{msg['content']}</div>
                        <div class="bubble-footer">{msg.get('time','')}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    rt  = msg.get("response_time", "")
                    wc  = word_count(msg["content"])
                    rt_badge  = f'<span class="speed-badge">⚡ {rt}s</span>' if rt else ""
                    wc_badge  = f'<span class="wc-badge">📝 {wc} words</span>'
                    st.markdown(f"""
                    <div class="chat-row">
                      <div class="av av-bot">⚡</div>
                      <div>
                        <div class="bubble bubble-bot">{msg['content']}</div>
                        <div class="bubble-footer">{rt_badge} {wc_badge} &nbsp; {msg.get('time','')}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ── Suggested questions ──────────────────────────
            st.markdown('<div class="sec-label">✨ Suggested Queries</div>', unsafe_allow_html=True)
            suggestions = [
                "What is the central thesis of this document?",
                "Summarise the key findings or conclusions.",
                "What are the main recommendations?",
                "List the most important data points or statistics.",
                "What problem does this document address?",
                "Are there any limitations or caveats mentioned?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    st.markdown('<span class="sug">', unsafe_allow_html=True)
                    if st.button(f"💡 {s}", key=f"sug_{i}", use_container_width=True):
                        st.session_state["prefill"] = s
                        st.rerun()
                    st.markdown('</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Input row ───────────────────────────────────────
        prefill = st.session_state.pop("prefill", "")
        user_q  = st.text_input(
            "Ask the Oracle",
            value=prefill,
            placeholder="e.g. What are the key conclusions of this document?",
            label_visibility="collapsed",
            key="oracle_input",
        )

        c_ask, c_opt = st.columns([2, 2])
        with c_opt:
            show_src = st.checkbox("📚 Show source evidence", value=True)
        with c_ask:
            ask_btn = st.button("🔮  Consult the Oracle", use_container_width=True,
                                disabled=not user_q)

        if ask_btn and user_q:
            ts = time.strftime("%H:%M")
            st.session_state.chat_history.append({"role": "user", "content": user_q, "time": ts})

            llm       = get_llm(model_name, temperature, max_tokens, GROQ_API_KEY)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": top_k})

            with st.spinner("🔮  The Oracle is consulting the document…"):
                try:
                    answer, elapsed = run_rag(user_q, retriever, llm)
                except Exception as e:
                    answer  = f"⚠️ Error: {e}"
                    elapsed = 0.0

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "response_time": elapsed, "time": ts,
            })
            st.session_state.total_queries    += 1
            st.session_state.response_times.append(elapsed)
            st.session_state.total_words_read += word_count(answer)

            if show_src:
                src_docs = retriever.invoke(user_q)
                with st.expander(f"📚  {len(src_docs)} Evidence Segments Retrieved"):
                    for i, doc in enumerate(src_docs, 1):
                        pg = doc.metadata.get("page", "?")
                        st.markdown(f"""
                        <div class="chunk-card">
                          <span class="chunk-tag">Segment {i} · Page {pg}</span>
                          {doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)
            st.rerun()


# ════════════════════════════════════════════════════════════
# TAB 3 — SESSION LOGBOOK
# ════════════════════════════════════════════════════════════
with tab_logbook:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📜</div>
          <div class="empty-title">The Logbook is empty</div>
          <div class="empty-sub">Your Oracle conversations will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        qa_pairs = []
        hist = st.session_state.chat_history
        for i, msg in enumerate(hist):
            if msg["role"] == "user":
                nxt = hist[i + 1] if i + 1 < len(hist) else None
                qa_pairs.append((msg, nxt))

        # Summary strip
        total_words = sum(word_count(p[1]["content"]) for p in qa_pairs if p[1])
        avg_spd = (sum(st.session_state.response_times) / len(st.session_state.response_times)
                   if st.session_state.response_times else 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Exchanges", len(qa_pairs))
        c2.metric("Words Generated", f"{total_words:,}")
        c3.metric("Avg Speed",       f"{avg_spd:.1f}s")
        st.markdown("---")

        for idx, (q, a) in enumerate(qa_pairs, 1):
            wc_str = f" · {word_count(a['content'])} words" if a else ""
            rt_str = f" · ⚡ {a['response_time']}s" if (a and a.get('response_time')) else ""
            with st.expander(f"#{idx} — {q['content'][:72]}{'…' if len(q['content'])>72 else ''}  ·  {q.get('time','')}{wc_str}{rt_str}"):
                st.markdown(f"""
                <div class="hist-q">
                  <div class="hist-label" style="color:var(--text-ghost);">▸ Question</div>
                  <div style="color:var(--text);font-size:0.9rem;">{q['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                if a:
                    rt = a.get("response_time", "")
                    st.markdown(f"""
                    <div class="hist-a">
                      <div class="hist-label" style="color:var(--cyan);">◈ Oracle Response</div>
                      <div style="color:var(--text-dim);font-size:0.88rem;line-height:1.75;">{a['content']}</div>
                      {"<div style='margin-top:0.6rem;font-family:IBM Plex Mono,monospace;font-size:0.68rem;color:var(--text-ghost);'>⚡ " + str(rt) + "s &nbsp;·&nbsp; " + str(word_count(a['content'])) + " words &nbsp;·&nbsp; " + reading_time(a['content']) + "</div>" if a else ""}
                    </div>
                    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ════════════════════════════════════════════════════════════
with tab_export:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📤</div>
          <div class="empty-title">Nothing to export yet</div>
          <div class="empty-sub">Complete a conversation in the <strong>Oracle Chat</strong> tab first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-label">Export Conversation</div>', unsafe_allow_html=True)

        # Build text export
        lines = [
            "═══════════════════════════════════════",
            "  LEXIS AI — Session Export",
            f"  Document : {st.session_state.doc_stats['filename'] or 'N/A'}",
            f"  Pages    : {st.session_state.doc_stats['pages']}",
            f"  Exported : {time.strftime('%Y-%m-%d %H:%M')}",
            "═══════════════════════════════════════\n",
        ]
        hist = st.session_state.chat_history
        for i, msg in enumerate(hist):
            if msg["role"] == "user":
                lines.append(f"[{msg.get('time','')}] YOU:\n{msg['content']}\n")
            else:
                rt = f"  ({msg.get('response_time','')}s)" if msg.get("response_time") else ""
                lines.append(f"[{msg.get('time','')}] LEXIS AI{rt}:\n{msg['content']}\n")
                lines.append("─" * 50 + "\n")

        export_text = "\n".join(lines)

        st.download_button(
            label="📥  Download as .txt",
            data=export_text,
            file_name=f"lexis_ai_session_{time.strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
        st.text_area("", value=export_text[:1500] + ("\n…[truncated]" if len(export_text) > 1500 else ""),
                     height=300, disabled=True, label_visibility="collapsed")
