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
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    )

    summarization_type = st.radio(
        "📝 Summarization Strategy",
        ["Stuff", "Map Reduce", "Refine"]
    )

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

if not GROQ_API_KEY:
    st.warning("⚠️ Please enter your GROQ API key in the sidebar.")
    st.stop()

# -------------------------------------------------------
# MODEL INITIALIZATION
# -------------------------------------------------------
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_token,
    streaming=False
)

st.sidebar.success("✅ Model Ready!")

# -------------------------------------------------------
# PROCESS PDF
# -------------------------------------------------------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    st.info(f"📄 Loaded {len(pages)} pages from PDF.")

    if st.button("🚀 Generate Summary"):

        with st.spinner("⏳ Generating summary..."):

            # -------------------------------------------------------
            # STUFF CHAIN
            # -------------------------------------------------------
            if summarization_type == "Stuff":

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                prompt = ChatPromptTemplate.from_template(
                    "Write a concise summary of the following document:\n\n{context}"
                )

                chain = (
                    {"context": format_docs}
                    | prompt
                    | model
                    | StrOutputParser()
                )

                result = chain.invoke(pages)

            # -------------------------------------------------------
            # MAP REDUCE
            # -------------------------------------------------------
            elif summarization_type == "Map Reduce":

                map_prompt = ChatPromptTemplate.from_template(
                    "Summarize this section:\n\n{section}"
                )

                map_chain = (
                    {"section": RunnablePassthrough()}
                    | map_prompt
                    | model
                    | StrOutputParser()
                )

                reduce_prompt = ChatPromptTemplate.from_template(
                    "Combine the following summaries into one coherent summary:\n\n{summaries}"
                )

                reduce_chain = (
                    {"summaries": lambda x: "\n".join(x)}
                    | reduce_prompt
                    | model
                    | StrOutputParser()
                )

                page_summaries = map_chain.batch(
                    [p.page_content for p in pages]
                )

                result = reduce_chain.invoke(page_summaries)

            # -------------------------------------------------------
            # REFINE
            # -------------------------------------------------------
            else:

                initial_prompt = ChatPromptTemplate.from_template(
                    "Write a concise summary of this section:\n\n{context}"
                )

                refine_prompt = ChatPromptTemplate.from_template(
                    "Existing summary:\n{existing_answer}\n\n"
                    "Refine it using the new context below:\n\n{context}"
                )

                parser = StrOutputParser()

                # First page
                current_summary = (
                    initial_prompt
                    | model
                    | parser
                ).invoke({"context": pages[0].page_content})

                # Refine with remaining pages
                for page in pages[1:]:
                    current_summary = (
                        refine_prompt
                        | model
                        | parser
                    ).invoke({
                        "existing_answer": current_summary,
                        "context": page.page_content
                    })

                result = current_summary

        # -------------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------------
        st.markdown("## 📌 Final Summary")
        st.success(result)

else:
    st.info("👆 Upload a PDF file to get started.")
