import streamlit as st
import os
import tempfile

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader , UnstructuredURLLoader
from langchain_community.tools import WikipediaQueryRun , ArxivQueryRun , DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper ,ArxivAPIWrapper
from langchain_core.tools import Tool


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="📄 AI PDF Summarizer",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center;'>🤖 AI PDF Summarizer</h1>
    <p style='text-align: center; font-size:18px;'>
        Upload a PDF and generate intelligent summaries using different strategies.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# SIDEBAR - MODEL SETTINGS
# -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    GROQ_API_KEY = st.text_input("🔑 GROQ API Key", type="password")

    temperature = st.slider("🌡 Temperature", 0.0, 1.0, 0.7, 0.1)
    max_token = st.slider("🧠 Max Tokens", 100, 4096, 1024)

    model_name = st.selectbox(
        "🤖 Model",
        ["Gemma2-9b-It","llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

    summarization_type = st.radio(
        "📝 Summarization Strategy",
        ["Stuff", "Map Reduce", "Refine"]
    )

wiki_wrapper =WikipediaAPIWrapper()

wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="Useful for when you need to answer questions about current events."
)


model = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_token,
    streaming=False
)

math_chain = LLMChain.from_llm(model)

math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for when you need to answer questions about math."
)

prompt= """
You are a helpful assistant with access to tools. Use a tool at most once unless absolutely necessary. 
If you already know the answer, respond directly. Do not loop tool calls.
input : {input}
"""

prompt_template = prompt_template(
    input_template=['input'],
    template =prompt
)
