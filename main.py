# main.py

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
import os
from dotenv import load_dotenv


load_dotenv()  # Load variables from .env file

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL: str = "gemini/gemini-2.0-flash" # Your .env loader and GEMINI_API_KEY

# Setup OpenAI-compatible Gemini client
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# Configure model
model = OpenAIChatCompletionsModel(
    model=GEMINI_MODEL,
    openai_client=external_client
)

# Create a very simple task-doing agent
agent = Agent(
    name="study_helper",
    instructions="You are a helpful personal study assistant. Answer clearly and concisely.",
    model=model
)

# RunConfig
config = RunConfig(
    model=model,
    model_provider=external_client,
    Agent=agent
)

# Run agent using Runner
if __name__ == "__main__":
    result = Runner.run_sync("How do I prepare for the Python interview?", config=config)
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting chat.")
        break

    result = Runner.run_sync(user_input, agent=agent, config=config)
    print(f"{agent.name.capitalize()}: {result.message.content}")
    print(result.message.content)

