"""
LangSmith Integration Test - ReAct Agent Demo
==============================================
This script demonstrates LangSmith tracing with a ReAct-style agent.

LangChain Version: v1.0+
Documentation Reference: https://docs.smith.langchain.com/tutorials/Developers/agents
Last Updated: October 2024

Prerequisites:
- OPENAI_API_KEY must be set in environment
- LANGSMITH_API_KEY must be set in environment
- LANGSMITH_TRACING must be set to "true"
- LANGSMITH_PROJECT can be set (optional, defaults to "default")
"""

import os
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent


# ============================================================================
# STEP 1: Environment Variable Verification
# ============================================================================
# Before we begin, we need to verify that all required API keys are set.
# Think of API keys like passwords - they authenticate our app with external services.

print("\n" + "="*70)
print("STEP 1: Verifying Environment Configuration")
print("="*70)

# Check if OpenAI API key exists (required for the LLM to work)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "❌ OPENAI_API_KEY environment variable must be set.\n"
        "   Set it with: export OPENAI_API_KEY=sk-..."
    )
print("✅ OpenAI API key is configured")

# Check if LangSmith API key exists (required for tracing/monitoring)
if not os.getenv("LANGSMITH_API_KEY"):
    print("⚠️  Warning: LANGSMITH_API_KEY not set. Tracing will not work.")
    print("   Set it with: export LANGSMITH_API_KEY=<your-key>")
else:
    print("✅ LangSmith API key is configured")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"   Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")


# ============================================================================
# STEP 2: Define Tools
# ============================================================================
# Tools are functions that our agent can use to get information or perform actions.
# Think of them like apps on your phone - the agent can "open" them when needed.

print("\n" + "="*70)
print("STEP 2: Defining Agent Tools")
print("="*70)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    This is a MOCK tool for demonstration purposes. In a real application,
    you would connect to an actual weather API (like OpenWeatherMap).
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the weather
    """
    # Mock weather data - in reality, this would come from an API call
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

print("✅ Weather tool defined")
print("   This tool allows the agent to check weather for specific cities")


# ============================================================================
# STEP 3: Create and Test the Agent
# ============================================================================
if __name__ == "__main__":
    print("\n🚀 Starting LangSmith Setup Demo")
    print("   Using the SIMPLIFIED create_agent approach (LangChain v1.0)!")
    
    print("\n" + "="*70)
    print("STEP 3: Creating and Testing Agent with LangSmith Tracing")
    print("="*70)
    
    # Create the agent - THIS IS IT! Just one simple call to create_agent()
    # It handles everything: model initialization, tool calling, state management
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Use the weather tool when asked about weather."
    )
    print("✅ Agent created with create_agent() - all setup done automatically!")
    
    # Define test cases to demonstrate different behaviors
    test_queries = [
        "What is the weather in Philadelphia?",  # Will use get_weather tool
        "What is the capital of France?",  # Will use LLM knowledge only
        "Can you check the weather in Seattle and San Francisco?",  # Multiple tool calls
    ]
    
    print(f"\n📝 Running {len(test_queries)} test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'─'*70}")
        print(f"Test {i}: {query}")
        print(f"{'─'*70}")
        
        try:
            # Invoke the agent with the user's question
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"test-{i}"}}
            )
            
            # Extract and display the final response
            final_message = result["messages"][-1]
            print(f"\n✓ Response: {final_message.content}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    # Display summary and next steps
    print(f"\n{'='*70}")
    print("✅ Testing Complete!")
    print("="*70)
    print(f"\n📊 View traces in LangSmith:")
    print(f"   → https://smith.langchain.com/")
    print(f"   → Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print("\n🔍 In LangSmith, you should see:")
    print("   • Full conversation history for each test")
    print("   • Tool invocations (get_weather calls)")
    print("   • LLM calls and responses")
    print("   • Token usage and latency metrics")
    print("   • Visual workflow execution graph")
    print("="*70)
    
    print("\n✨ Demo completed successfully!\n")

