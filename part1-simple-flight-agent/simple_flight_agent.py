from langchain.agents import create_agent

# =============================================================================
# TOOL DEFINITION: get_flight_status
# =============================================================================
# This is our "tool" - think of it as a special function that our AI agent 
# can call when it needs to look up flight information.
# 
# In a real-world scenario, this would connect to an actual flight API.
# For learning purposes, we're using mock (fake) data.
#
# The docstring (text in triple quotes) is IMPORTANT - the AI reads this to 
# understand what the tool does and when to use it!
# =============================================================================

def get_flight_status(flight_number: str) -> str:
    """Gets the status for a given flight number."""
    # This is a mock response for demonstration
    # In production, you'd call a real flight API here
    return f"Flight {flight_number} is on time."


# =============================================================================
# AGENT CREATION
# =============================================================================
# Now we create our AI agent. Think of an agent as an AI assistant that can:
# 1. Understand what you're asking (natural language processing)
# 2. Decide which tools to use (reasoning)
# 3. Execute those tools (function calling)
# 4. Give you a helpful response (generation)
#
# Parameters explained:
# - model: Which AI model to use (GPT-4o is OpenAI's latest and greatest)
# - tools: A list of functions the agent can call when needed
# - system_prompt: Instructions that tell the agent its role and behavior
# =============================================================================

agent = create_agent(
    model="openai:gpt-4o",  # Using OpenAI's GPT-4o model
    tools=[get_flight_status],
    system_prompt="You are a helpful flight assistant.",
)


# =============================================================================
# RUNNING THE AGENT
# =============================================================================
# The agent.invoke() method sends a message to our agent and gets a response.
#
# We structure the input as a dictionary with a "messages" key containing
# a list of message objects (each with a role and content).
#
# This format mimics how chatbots work - maintaining a conversation history.
# =============================================================================

# Let's run it!
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of flight AA123?"}]}
)

# Print the entire response to see what the agent returns
print(response)

