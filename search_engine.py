import streamlit as st 
from langchain_community.tools import WikipediaQueryRun , ArxivQueryRun , DuckDuckGoSearchResults
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper ,ArxivAPIWrapper
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader ,PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import AgentType , initialize_agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

api = WikipediaAPIWrapper(top_k_results=1, doc_context_chars_mx =250)
wiki =  WikipediaQueryRun(api_wrapper =api)
arxiv = ArxivAPIWrapper(top_k_results=1, doc_context_chars_mx =250)
arxiv = ArxivQueryRun(api_wrapper =arxiv)

search =DuckDuckGoSearchResults(name="Search", num_results=2)

tools = [wiki, arxiv]

with st.sidebar:
    st.header("Model Settings")
    LANGCHAIN_API_KEY = st.text_input("LANGCHAIN_API_KEY")
    GROQ_API_KEY = st.text_input("GROQ_API_KEY")
    HF_TOKEN = st.text_input("HF_API_KEY")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_token = st.slider("Max Tokens", 1, 2048, 512)
    model_name = st.selectbox("Model Name", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

# Load secrets into environment
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "PDF ChatBot"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN
st.title("📄 Location PDF Chat History")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) 

prompt= st.chat_input(placeholder="What is the Machine Learnig") 
if prompt and GROQ_API_KEY and model_name and temperature and max_token and HF_TOKEN:    
    st.session_state.messages.append({'role':"user","content":prompt})
    st.chat_message("user").write(prompt)
    model = ChatGroq(
        model_name=model_name,
        temperature=temperature,    max_tokens=max_token ,
        api_key=GROQ_API_KEY , streaming =False)

    tools =[search,wiki]#,arxiv

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful assistant with access to tools. "
        "Use a tool at most once unless absolutely necessary. "
        "If you already know the answer, respond directly. "
        "Do not loop tool calls."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(model, tools, agent_prompt)
    # = AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_tool_error = True)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        early_stopping_method="generate",
        max_execution_time=20,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt})
            st.write(response["output"])
            #st.write(response)
            st.session_state.messages.append({'role':"assistant","content":response["output"]})