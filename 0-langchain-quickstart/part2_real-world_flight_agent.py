SYSTEM_PROMPT = """Your goal is to provide passengers with accurate and helpful flight information. You should be professional, but with a friendly and slightly witty tone.

You have access to two tools:
- get_flight_details: Use this to get the status, gate, and departure time for a specific flight number.
- get_user_home_airport: If a user asks about flights from 'my airport' or 'home', use this to find their registered home airport.

Always confirm the flight number before providing details. If a user asks a general question, use their home airport to provide relevant examples."""

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# A tool for specific flight details
@tool
def get_flight_details(flight_number: str) -> str:
    """Gets flight details like status, gate, and time for a given flight number."""
    # Mock data for the example
    return f"Flight {flight_number} is on time, departing from Gate B12 at 8:45 PM."

# Define the data structure for our runtime context
@dataclass
class Context:
    """Custom runtime context schema for user info."""
    user_id: str
    
@dataclass
class ResponseFormat:
    """Response schema for the flight agent."""
    friendly_response: str
    flight_details: str | None = None

# A tool that uses the runtime context to find the user's home airport
@tool
def get_user_home_airport(runtime: ToolRuntime[Context]) -> str:
    """Retrieves the user's home airport based on their user ID."""
    user_id = runtime.context.user_id
    # In a real app, you'd look this up in a database
    return "JFK" if user_id == "user_abc" else "SFO"


model = init_chat_model(
    "openai:gpt-4o",
    temperature=0.3, # Leans toward more factual responses
    timeout=10,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_home_airport, get_flight_details],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "flight_convo_1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the status of my flight home today?"}]},
    config=config,
    context=Context(user_id="user_abc")
)
print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "It's flight UA456."}]},
    config=config,
    context=Context(user_id="user_abc")
)
print(response['structured_response'])