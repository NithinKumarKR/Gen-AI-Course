import streamlit as st
import os
import time
import tempfile
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
load_dotenv()




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


model = ChatGroq(
    model_name=model_name,
    temperature=temperature,    max_tokens=max_token ,
    api_key=os.getenv("GROQ_API_KEY") )



embeddings = HuggingFaceEndpointEmbeddings(
    task="feature-extraction",
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided.
Be concise and clear.

<context>
{context}
</context>

Question: {input}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create vector DB
def create_vector_db():
    loader = PyPDFDirectoryLoader("source_pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = splitter.split_documents(docs)

    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

# UI

if st.button("Create Vector DB"):
    create_vector_db()
    st.success("Vector database ready!")

user_question = st.text_input("Ask a question")


if user_question and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    start = time.time()
    answer = rag_chain.invoke(user_question)
    end = time.time()

    st.write("### Answer:")
    st.write(answer)
    st.write(f"⏱ Time taken: {end - start:.2f} sec")

    with st.expander("📚 Retrieved Documents"):
        docs = retriever.invoke(user_question)
        for doc in docs:
            st.write(doc.page_content)
            st.write("-" * 50)
