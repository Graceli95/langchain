"""
LangSmith Integration Test - ReAct Agent
=========================================
This script demonstrates LangSmith tracing with a ReAct-style agent.

A ReAct agent combines:
- REASONING: Thinking about what action to take
- ACTING: Using tools or answering directly

Think of it like a smart assistant that can both answer questions from memory
AND use tools (like checking the weather) when needed.

LangChain Version: v1.0+
Documentation Reference: https://docs.langchain.com/oss/python/langgraph
Last Updated: October 2024

Prerequisites:
- OPENAI_API_KEY must be set in environment
- LANGSMITH_API_KEY must be set in environment
- LANGSMITH_TRACING=true must be set
- LANGSMITH_PROJECT should be set (e.g., "my-first-agent")
"""

import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


# ============================================================================
# STEP 1: Verify Environment Configuration
# ============================================================================
# Before we start, let's make sure all required API keys are configured.
# Think of API keys like passwords that let our code talk to external services.

print("\n" + "="*70)
print("STEP 1: Verifying Environment Configuration")
print("="*70)

# Check for OpenAI API Key (required for the LLM)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "[ERROR] OPENAI_API_KEY environment variable must be set.\n"
        "   This key allows us to use OpenAI's GPT models."
    )
else:
    print("[OK] OpenAI API key is configured")

# Check for LangSmith Configuration (required for tracing)
if not os.getenv("LANGSMITH_API_KEY"):
    print("[WARNING] LANGSMITH_API_KEY not set. Tracing will not work.")
    print("   You can still run the agent, but won't see traces in LangSmith.")
else:
    print("[OK] LangSmith API key is configured")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"   Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")

print("="*70 + "\n")


# ============================================================================
# STEP 2: Define Tools (The Agent's Abilities)
# ============================================================================
# Tools are like giving the AI special abilities. In this case, we're giving it
# the ability to check the weather. The @tool decorator tells LangChain this
# function can be used by the agent.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    This is a MOCK tool - it returns fake weather data for demonstration.
    In a real application, this would call an actual weather API.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the weather conditions
    """
    # Mock weather data (pretend API response)
    # In production, you'd call a real weather API like OpenWeatherMap
    weather_data = {
        "philadelphia": "Sunny and 72°F with light winds",
        "new york": "Cloudy with a chance of rain, 65°F",
        "san francisco": "Foggy and cool, 58°F",
        "seattle": "Rainy as always, 55°F",
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"The weather in {city} is: {weather_data[city_lower]}"
    else:
        return f"Weather data for {city} is not available. It's probably nice though!"


# ============================================================================
# STEP 3: Define State Schema (The Agent's Memory)
# ============================================================================
# State is like the agent's short-term memory. It keeps track of the conversation
# so far. The `messages` field stores all messages (user queries, AI responses,
# and tool outputs).
#
# Think of it like a chat history that gets passed between different parts of
# the agent workflow.

class AgentState(TypedDict):
    """State schema for the agent workflow.
    
    This defines what information flows through the agent as it processes a query.
    The Annotated[list, add_messages] means messages will be automatically
    appended (not replaced) as the agent runs.
    """
    messages: Annotated[list, add_messages]


# ============================================================================
# STEP 4: Define Decision Logic (The Agent's Brain)
# ============================================================================
# This function decides what the agent should do next:
# - If the LLM wants to use a tool → route to "tools"
# - If the LLM has a final answer → route to "end"

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or end.
    
    This is the "routing function" - it decides the next step based on what
    the LLM just said. It's like a traffic controller directing the workflow.
    
    Args:
        state: Current agent state containing all messages
        
    Returns:
        "tools" if agent wants to use tools, "end" if ready to respond
    """
    last_message = state["messages"][-1]
    
    # Check if the LLM made a tool call
    # (e.g., decided to check the weather instead of answering directly)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, the LLM has the final answer, so we're done
    return "end"


# ============================================================================
# STEP 5: Define Model Calling Function (The Agent's Decision Maker)
# ============================================================================
# This function calls the LLM (Large Language Model) to decide what to do.
# The LLM can either:
# 1. Decide to use a tool (e.g., check weather)
# 2. Answer directly using its knowledge (e.g., "Paris is the capital of France")

def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state.
    
    This is where the AI "thinks" about what to do. We give it:
    - A system prompt (instructions on how to behave)
    - The conversation history
    - Access to tools it can use
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with the LLM's response (either a tool call or final answer)
    """
    # Initialize the LLM (GPT-4o-mini is a fast, affordable model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to the LLM (let it know what tools are available)
    tools = [get_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a system prompt (instructions for the AI)
    # This is like giving an employee a job description
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "For weather-related questions, use the get_weather tool. "
            "For general questions, answer using your knowledge."
        )
    }
    
    # Call the model with the full conversation history
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Return the updated state with the LLM's response
    return {"messages": [response]}


# ============================================================================
# STEP 6: Build the LangGraph Workflow (Connect Everything Together)
# ============================================================================
# This creates the actual workflow graph. Think of it like a flowchart:
# 1. Start at "agent" node (LLM decides what to do)
# 2. If tool needed → go to "tools" node → back to "agent"
# 3. If answer ready → END

def create_agent():
    """Create and compile the ReAct agent workflow.
    
    This builds the workflow graph that defines how the agent operates:
    - Nodes: The different "stations" the workflow can visit
    - Edges: The "roads" between stations
    - Conditional Edges: "Forks in the road" where we choose which way to go
    
    Returns:
        Compiled LangGraph application ready to process queries
    """
    print("Building agent workflow...")
    
    # Initialize the state graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes (the different processing steps)
    workflow.add_node("agent", call_model)  # The LLM decision-maker
    workflow.add_node("tools", ToolNode([get_weather]))  # Tool execution
    
    # Set the entry point (where the workflow starts)
    workflow.set_entry_point("agent")
    
    # Add conditional edges (decision points in the workflow)
    # After "agent" node, decide whether to use tools or end
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If tools needed, go to tools node
            "end": END,        # If done, end the workflow
        }
    )
    
    # Add edge from tools back to agent
    # After using a tool, go back to the agent to process the result
    workflow.add_edge("tools", "agent")
    
    # Compile the graph into a runnable application
    print("[OK] Agent workflow compiled successfully!\n")
    return workflow.compile()


# ============================================================================
# STEP 7: Test the Agent (Put It Through Its Paces)
# ============================================================================
# Now let's test our agent with different types of questions to make sure
# it can handle both tool-based queries and general knowledge questions.

def test_agent():
    """Run test queries to verify the agent and LangSmith integration.
    
    We'll test three scenarios:
    1. Weather question (should use tool)
    2. General knowledge (should answer directly)
    3. Multiple weather queries (should use tool multiple times)
    """
    
    print("="*70)
    print("Testing ReAct Agent with LangSmith Tracing")
    print("="*70)
    
    # Create the agent
    agent = create_agent()
    
    # Define test cases that cover different agent behaviors
    test_queries = [
        "What is the weather in Philadelphia?",           # Tool use
        "What is the capital of France?",                  # Direct answer
        "Can you check the weather in Seattle and San Francisco?",  # Multiple tools
    ]
    
    # Run each test query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}: {query}")
        print(f"{'-'*70}")
        
        try:
            # Invoke the agent with the query
            # The agent will:
            # 1. Receive the question
            # 2. Decide whether to use a tool or answer directly
            # 3. Execute tools if needed
            # 4. Return the final answer
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"test-{i}"}}
            )
            
            # Extract and display the final response
            final_message = result["messages"][-1]
            print(f"\n[SUCCESS] Response: {final_message.content}")
            
        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}")
    
    # Print summary and next steps
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print("="*70)
    print(f"\nView detailed traces in LangSmith:")
    print(f"   URL: https://smith.langchain.com/")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print("\nIn LangSmith, you should see:")
    print("   * Full conversation history for each test")
    print("   * Tool invocations (when get_weather was called)")
    print("   * LLM calls and responses")
    print("   * Token usage and latency metrics")
    print("   * A visual graph showing the workflow execution path")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
# This is where the script starts running when you execute it.

if __name__ == "__main__":
    test_agent()

