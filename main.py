import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompts = ChatPromptTemplate.from_messages(
    [
        ("system","your are a helpful assistant.Please responsed to the question asked"),
        ("human","Question:{question}")
    ]
)


st.title("Langchain Demo with Gemma")

input_text= st.text_input("what are your thinking in mind")

llm = Ollama(model="gemma:2b")

output_parser = StrOutputParser()

chain  =prompts | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
