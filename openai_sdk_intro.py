from dotenv import load_dotenv
from agents import Agent, Runner, trace
import asyncio

load_dotenv(override=True)

jokes_agent = Agent(name="Jokester", instructions="You are a joke teller", model="gpt-4o-mini")

async def get_joke():
    with trace("Telling a joke"):
        result = await Runner.run(jokes_agent, "Tell a joke about Autonomous AI Agents")
        print(result.final_output)


asyncio.run(get_joke())