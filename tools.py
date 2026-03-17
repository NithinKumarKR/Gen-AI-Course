import os
from dotenv import load_dotenv
load_dotenv()

from crewai_tools import YoutubeVideoSearchTool
from langchain_groq import ChatGroq

# Initialize Groq LLM
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# YouTube search tool with Groq and Hugging Face
youtube_search_tool = YoutubeVideoSearchTool(
    config=dict(
        llm=dict(
            provider="groq",
            config=dict(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model_name="all-MiniLM-L6-v2",
                # Local embeddings - no API key needed
            ),
        ),
    )
)
