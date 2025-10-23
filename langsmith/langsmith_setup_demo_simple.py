"""
LangSmith Integration Test - Simple ReAct Agent (Using Built-in create_agent)
==============================================================================
This is the SIMPLIFIED version using LangChain's built-in create_agent function.

This demonstrates that you DON'T need to manually build the entire workflow
when you're doing standard agent creation. LangChain provides a helper function!

LangChain Version: v1.0+
Documentation Reference: https://docs.langchain.com/oss/python/langchain/agents
Last Updated: October 2024

Prerequisites:
- OPENAI_API_KEY must be set in environment
- LANGSMITH_API_KEY must be set in environment (optional, for tracing)
- LANGSMITH_TRACING=true must be set (optional, for tracing)
"""

import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# ============================================================================
# STEP 1: Verify Environment Configuration
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Verifying Environment Configuration")
print("="*70)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "[ERROR] OPENAI_API_KEY environment variable must be set.\n"
        "   This key allows us to use OpenAI's GPT models."
    )
else:
    print("[OK] OpenAI API key is configured")

if not os.getenv("LANGSMITH_API_KEY"):
    print("[WARNING] LANGSMITH_API_KEY not set. Tracing will not work.")
else:
    print("[OK] LangSmith API key is configured")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"   Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")

print("="*70 + "\n")


# ============================================================================
# STEP 2: Define Tools
# ============================================================================
# Tools are functions the agent can use. The @tool decorator registers them
# with LangChain so the LLM knows they exist and how to call them.

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
# STEP 3: Create Agent and Test It
# ============================================================================
def test_agent():
    """Create and test the agent."""
    
    print("="*70)
    print("Testing Simple ReAct Agent with create_agent()")
    print("="*70)
    
    # STEP 3A: Create the agent (THE SIMPLE WAY!)
    # ============================================
    # Instead of manually building a StateGraph with nodes, edges, and 
    # conditional routing, we just use create_agent() and let LangChain 
    # handle all that for us!
    #
    # It's like ordering a car vs. building one from parts - both get you 
    # driving, but one is way faster for standard use cases!
    
    print("Creating agent using built-in create_agent()...")
    
    # Option 1: Pass a model string (simplest - LangChain figures out the provider)
    # agent = create_agent(
    #     model="openai:gpt-4o-mini",
    #     tools=[get_weather],
    #     system_prompt="You are a helpful AI assistant."
    # )
    
    # Option 2: Pass a model instance (more control over parameters)
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0  # Low temperature for consistent responses
    )
    
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "You are a helpful AI assistant. "
            "For weather-related questions, use the get_weather tool. "
            "For general questions, answer using your knowledge."
        )
    )
    
    print("[OK] Agent created successfully!\n")
    
    # STEP 3B: Test the agent with different queries
    # ===============================================
    
    # Test cases
    test_queries = [
        "What is the weather in Philadelphia?",
        "What is the capital of France?",
        "Can you check the weather in Seattle and San Francisco?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"Test {i}: {query}")
        print(f"{'-'*70}")
        
        try:
            # Invoke the agent
            # We pass the query as a simple string - create_agent handles the rest!
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"test-{i}"}}
            )
            
            # Extract final response
            final_message = result["messages"][-1]
            print(f"\n[SUCCESS] Response: {final_message.content}")
            
        except Exception as e:
            print(f"\n[ERROR] Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print("="*70)
    print(f"\nView detailed traces in LangSmith:")
    print(f"   URL: https://smith.langchain.com/")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print("="*70 + "\n")


# ============================================================================
# COMPARISON: Simple vs. Manual Approach
# ============================================================================
"""
SIMPLE APPROACH (this file):
-----------------------------
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

✅ Pros:
- 4 lines of code instead of 80+
- Less room for bugs
- Easier to read and maintain
- Perfect for standard use cases

❌ Cons:
- Less control over the exact workflow
- Harder to add custom logic between steps
- Can't easily visualize what's happening under the hood


MANUAL APPROACH (langsmith_setup_demo.py):
------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode([get_weather]))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {...})
workflow.add_edge("tools", "agent")
agent = workflow.compile()

✅ Pros:
- Full control over the workflow
- Can add custom nodes (e.g., memory, validation, logging)
- Educational - you see exactly how agents work
- Can implement complex multi-agent patterns
- Can add cycles, human-in-the-loop, etc.

❌ Cons:
- Much more code
- More places for bugs
- Overkill for simple agents


WHEN TO USE WHICH?
------------------
Use create_agent() when:
- Building a standard ReAct agent
- You want quick prototyping
- You don't need custom workflow logic

Use manual StateGraph when:
- You need custom nodes or logic
- Building complex multi-agent systems
- You want to understand how agents work internally
- You need special routing or cycles in your workflow
"""


if __name__ == "__main__":
    test_agent()

