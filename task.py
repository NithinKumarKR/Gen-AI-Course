from crewai import Task
from tools import youtube_search_tool
from agents import blog_researcher, blog_writer

# Create a task for the senior blog content researcher
research_task = Task(
    description="Search YouTube for relevant videos on the topic {topic} and extract key insights. "
                "Summarize the most important points and findings.",
    expected_output="A comprehensive summary of relevant YouTube content for the topic {topic} with key insights and points",
    agent=blog_researcher,
    tools=[youtube_search_tool],
)

# Create a task for the senior blog content writer
writer_task = Task(
    description="Write a compelling, well-structured blog post for the topic {topic} "
                "based on the research findings. Make it engaging and informative.",
    expected_output="A complete, well-written blog post on {topic} with proper structure, introduction, body, and conclusion",
    agent=blog_writer,
    tools=[youtube_search_tool],
    async_execution=False,
    output_file="blog_post.md",
)
