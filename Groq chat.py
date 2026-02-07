import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()



with st.sidebar:
    st.header("Model Settings")
    LANGCHAIN_API_KEY = st.text_input("LANGCHAIN_API_KEY")
    GROQ_API_KEY = st.text_input("GROQ_API_KEY")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_token = st.slider("Max Tokens", 1, 2048, 512)
    model_name = st.selectbox("Model Name", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])



# Load secrets into environment
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ChatBot"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(
    model_name=model_name,  # recommended Groq model
    temperature=temperature,
    max_tokens=max_token
)

st.title("GROQ ChatBot")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | model | StrOutputParser()

question = st.text_input("Ask a question:")

if question:
    response = chain.invoke({"question": question})
    st.write(response)
