from crewai import Crew
from agents import blog_researcher, blog_writer
from tasks import research_task, writer_task
import os
from dotenv import load_dotenv

load_dotenv()

# Create the crew with Groq LLM and Hugging Face embeddings
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writer_task],
    verbose=True,
    memory=True,
    embedder={
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
            # No API key needed for local Hugging Face embeddings
        }
    }
)

# Run the crew with your topic
if __name__ == "__main__":
    result = crew.kickoff(inputs={'topic': 'Generative AI'})
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(result)
