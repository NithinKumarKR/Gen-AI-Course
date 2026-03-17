from crewai import Agent
from tools import youtube_search_tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# Create a senior blog content researcher agent
blog_researcher = Agent(
    role="Blog Research from YouTube Videos",
    goal="Get the relevant content for the topic {topic} from YouTube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning and Gen AI. "
        "You excel at extracting key insights from video content and summarizing them effectively."
    ),
    tools=[youtube_search_tool],
    llm=llm,
    allow_delegation=True,
    max_iter=5,
)

# Creating a senior blog writer agent
blog_writer = Agent(
    role="Senior Blog Content Writer",
    goal="Write a compelling blog post for the topic {topic} based on research",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in writing engaging blog posts in AI, Data Science, Machine Learning and Gen AI. "
        "You create well-structured, informative content that captivates readers."
    ),
    tools=[youtube_search_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=5,
)
