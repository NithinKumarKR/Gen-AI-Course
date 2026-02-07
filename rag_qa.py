import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader ,PyPDFDirectoryLoader

load_dotenv()

st.title("📄 RAG PDF Q&A")

# Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Model
model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

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
def create_vector_db(uploaded_file):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_docs = splitter.split_documents(docs)

    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

# UI
user_question = st.text_input("Ask a question")

if st.button("Create Vector DB"):
    create_vector_db(uploaded_file)
    st.success("Vector database ready!")

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
