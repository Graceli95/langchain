# Part 1: Your First, Simple Agent (Flights Edition)
# ===================================================
# This is our introduction to LangChain agents. An agent is like a smart assistant
# that can understand natural language questions and decide which tools to use to answer them.

# WHAT IS AN AGENT?
# Think of an agent as a helpful employee who knows:
# 1. What tools are available (like a toolbox)
# 2. When to use each tool (like knowing when to use a hammer vs. a screwdriver)
# 3. How to communicate the results back to you in a natural way

from langchain.agents import create_agent

# DEFINING A TOOL
# A "tool" is simply a Python function that the agent can call.
# The @tool decorator (commented out here) tells LangChain that this function
# is available for the agent to use when needed.
# @tool
def get_flight_status(flight_number: str) -> str:
    """Gets the status for a given flight number."""
    # This is a mock response for demonstration purposes
    # In a real application, this would call an actual flight tracking API
    return f"Flight {flight_number} is on time."

# CREATING THE AGENT
# Here we bring together three essential components:
# 1. MODEL: The AI brain (GPT-4o) that understands language and makes decisions
# 2. TOOLS: The functions the agent can use (just get_flight_status for now)
# 3. SYSTEM_PROMPT: Instructions that define the agent's personality and role
agent = create_agent(
    model="openai:gpt-4o",  # Using OpenAI's GPT-4o model as the "brain"
    tools=[get_flight_status],  # List of tools available to this agent
    system_prompt="You are a helpful flight assistant.",  # The agent's role/personality
)

# RUNNING THE AGENT
# The agent.invoke() method sends a message to the agent and gets a response.
# The agent will:
# 1. Read the user's question
# 2. Realize it needs to use the get_flight_status tool
# 3. Call that tool with "AA123" as the flight number
# 4. Return a natural language response based on the tool's output
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of flight AA123?"}]}
)

print(response)

