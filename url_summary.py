import streamlit as st
import os
import validators

# LangChain Imports
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="AI URL Summarizer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI URL Summarizer")
st.markdown("Generate intelligent summaries from any URL.")

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
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma-7b-it"
        ]
    )

    summarization_type = st.radio(
        "📝 Summarization Strategy",
        ["Stuff", "Map Reduce", "Refine"]
    )

# -------------------------------------------------------
# URL INPUT
# -------------------------------------------------------
url = st.text_input("Enter URL")

if not GROQ_API_KEY:
    st.warning("⚠️ Please enter your GROQ API key in the sidebar.")
    st.stop()

if not url:
    st.info("👆 Enter a URL to begin.")
    st.stop()

if not validators.url(url):
    st.error("❌ Please enter a valid URL.")
    st.stop()

# -------------------------------------------------------
# MODEL INITIALIZATION
# -------------------------------------------------------
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_token,
)

parser = StrOutputParser()

# -------------------------------------------------------
# GENERATE SUMMARY
# -------------------------------------------------------
if st.button("🚀 Generate Summary"):

    with st.spinner("⏳ Loading content..."):

        try:
            # Choose loader
            if "youtube.com" in url or "youtu.be" in url:
                loader = YoutubeLoader.from_youtube_url(url)
            else:
                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

            docs = loader.load()

            if not docs:
                st.error("No content could be extracted from this URL.")
                st.stop()

        except Exception as e:
            st.error(f"Error loading URL: {e}")
            st.stop()

    # -------------------------------------------------------
    # SPLIT DOCUMENTS (IMPORTANT FOR LARGE URLS)
    # -------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(docs)

    with st.spinner("🧠 Generating summary..."):

        # -------------------------------------------------------
        # STUFF STRATEGY
        # -------------------------------------------------------
        if summarization_type == "Stuff":

            def join_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_template(
                "Write a concise and clear summary of the following content:\n\n{context}"
            )

            chain = (
                RunnableLambda(lambda x: {"context": join_docs(x)})
                | prompt
                | model
                | parser
            )

            result = chain.invoke(split_docs)

        # -------------------------------------------------------
        # MAP REDUCE STRATEGY
        # -------------------------------------------------------
        elif summarization_type == "Map Reduce":

            map_prompt = ChatPromptTemplate.from_template(
                "Summarize this section clearly:\n\n{context}"
            )

            reduce_prompt = ChatPromptTemplate.from_template(
                "Combine the following summaries into one final coherent summary:\n\n{context}"
            )

            map_chain = map_prompt | model | parser
            reduce_chain = reduce_prompt | model | parser

            # Map step
            summaries = map_chain.batch(
                [{"context": doc.page_content} for doc in split_docs]
            )

            # Reduce step
            result = reduce_chain.invoke({
                "context": "\n\n".join(summaries)
            })

        # -------------------------------------------------------
        # REFINE STRATEGY
        # -------------------------------------------------------
        else:

            initial_prompt = ChatPromptTemplate.from_template(
                "Write a concise summary of:\n\n{context}"
            )

            refine_prompt = ChatPromptTemplate.from_template(
                "Existing summary:\n{existing_answer}\n\n"
                "Refine it using the new content below:\n\n{context}"
            )

            # First chunk
            current_summary = (
                initial_prompt
                | model
                | parser
            ).invoke({"context": split_docs[0].page_content})

            # Refine remaining chunks
            for doc in split_docs[1:]:
                current_summary = (
                    refine_prompt
                    | model
                    | parser
                ).invoke({
                    "existing_answer": current_summary,
                    "context": doc.page_content
                })

            result = current_summary

    # -------------------------------------------------------
    # DISPLAY RESULT
    # -------------------------------------------------------
    st.markdown("## 📌 Final Summary")
    st.success(result)
