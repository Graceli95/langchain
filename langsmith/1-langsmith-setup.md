# LangSmith Setup and Integration Guide

## Overview

**LangSmith** is a powerful platform for debugging, testing, evaluating, and monitoring LangChain applications. It provides:
- **Tracing**: Visualize every step of your LLM application's execution
- **Debugging**: Identify issues in complex agent workflows
- **Evaluation**: Test and benchmark your application's performance
- **Monitoring**: Track production usage and costs

This guide walks you through setting up LangSmith integration and building a test agent to verify everything works correctly.

---

## Prerequisites

Before starting, ensure you have:
- ✅ A LangSmith account (sign up at [smith.langchain.com](https://smith.langchain.com))
- ✅ `OPENAI_API_KEY` already set in your environment
- ✅ Python 3.9+ installed

---

## Step 1: Create Your LangSmith API Key

1. Navigate to [smith.langchain.com](https://smith.langchain.com)
2. Sign in to your account
3. Go to **Settings** → **API Keys**
4. Click **"Create API Key"**
5. Give it a descriptive name (e.g., "Development Key")
6. **Set an expiry date** (recommended: 30-90 days for development keys)
   - This limits the risk window if your key is accidentally exposed
   - You can always create a new key when it expires
   - For production, consider shorter expiry periods with automated rotation
7. Copy the generated API key and **store it securely** (you won't be able to see it again)

---

## Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install langchain langchain-openai langsmith langgraph
```

**Package breakdown:**
- `langchain`: Core LangChain framework
- `langchain-openai`: OpenAI integration for LangChain
- `langsmith`: LangSmith client for tracing and monitoring
- `langgraph`: Modern workflow framework (replaces deprecated LCEL)

---

## Step 3: Configure Environment Variables

Set up your environment to enable LangSmith tracing:

### On macOS/Linux:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_PROJECT=my-first-agent
```

### On Windows (PowerShell):
```powershell
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
$env:LANGSMITH_API_KEY="<your-api-key>"
$env:LANGSMITH_PROJECT="my-first-agent"
```

### On Windows (Command Prompt):
```cmd
set LANGSMITH_TRACING=true
set LANGSMITH_ENDPOINT=https://api.smith.langchain.com
set LANGSMITH_API_KEY=<your-api-key>
set LANGSMITH_PROJECT=my-first-agent
```

**Environment Variable Explanations:**
- `LANGSMITH_TRACING`: Enables automatic tracing of all LangChain operations
- `LANGSMITH_ENDPOINT`: The LangSmith API endpoint (usually the default shown above)
- `LANGSMITH_API_KEY`: Your unique API key for authentication
- `LANGSMITH_PROJECT`: Project name for organizing traces (you can use any name)

---

## Step 4: Understanding ReAct-Style Agents

A **ReAct (Reasoning + Acting)** agent is an intelligent system that:

1. **Reasons**: Thinks about what action to take based on the user's query
2. **Acts**: Decides whether to:
   - Use a tool (e.g., weather API, calculator, search)
   - Answer directly using the LLM's knowledge
3. **Observes**: Analyzes the result and decides if more actions are needed

### Architecture:
```
User Query → Agent Reasoning → Decision
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
              Use Tool(s)                  LLM Response
                    ↓                           ↓
              Tool Output                  Final Answer
                    ↓                           
              Final Answer                      
```

---

## Step 5: Build and Test Your Agent

The following Python script creates a ReAct-style agent with:
- A **weather tool** for weather-related questions
- An **LLM** for general knowledge queries
- **LangSmith tracing** to monitor execution

### Complete Test Script:

```python
"""
LangSmith Integration Test - ReAct Agent
=========================================
This script tests LangSmith tracing with a simple ReAct-style agent.

LangChain Version: v1.0+
Documentation Reference: https://docs.langchain.com/oss/python/langgraph
Last Updated: October 2024

Prerequisites:
- OPENAI_API_KEY must be set in environment
- LangSmith environment variables configured
"""

import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Verify LangSmith is configured
if not os.getenv("LANGSMITH_API_KEY"):
    print("⚠️  Warning: LANGSMITH_API_KEY not set. Tracing will not work.")
else:
    print("✅ LangSmith configuration detected")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"   Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")


# ============================================================================
# STEP 1: Define Tools
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the weather
    """
    # Mock weather data for demonstration
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
# STEP 2: Define State Schema
# ============================================================================

class AgentState(TypedDict):
    """State schema for the agent workflow."""
    messages: Annotated[list, add_messages]


# ============================================================================
# STEP 3: Create Agent Node Functions
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or end.
    
    Args:
        state: Current agent state
        
    Returns:
        "tools" if agent wants to use tools, "end" otherwise
    """
    last_message = state["messages"][-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the workflow
    return "end"


def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current state.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with LLM response
    """
    # Initialize the LLM with tool binding
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    # Get system prompt
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "For weather-related questions, use the get_weather tool. "
            "For general questions, answer using your knowledge."
        )
    }
    
    # Call the model
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


# ============================================================================
# STEP 4: Build the LangGraph Workflow
# ============================================================================

def create_agent():
    """Create and compile the ReAct agent workflow.
    
    Returns:
        Compiled LangGraph application
    """
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([get_weather]))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# STEP 5: Test the Agent
# ============================================================================

def test_agent():
    """Run test queries to verify the agent and LangSmith integration."""
    
    print("\n" + "="*70)
    print("Testing ReAct Agent with LangSmith Tracing")
    print("="*70)
    
    # Create the agent
    agent = create_agent()
    
    # Test cases
    test_queries = [
        "What is the weather in Philadelphia?",
        "What is the capital of France?",
        "Can you check the weather in Seattle and San Francisco?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}: {query}")
        print(f"{'─'*70}")
        
        try:
            # Invoke the agent
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"test-{i}"}}
            )
            
            # Extract final response
            final_message = result["messages"][-1]
            print(f"\n✓ Response: {final_message.content}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print("="*70)
    print(f"\n📊 View traces in LangSmith:")
    print(f"   → https://smith.langchain.com/")
    print(f"   → Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print("\nIn LangSmith, you should see:")
    print("   • Full conversation history")
    print("   • Tool invocations (get_weather)")
    print("   • LLM calls and responses")
    print("   • Token usage and latency metrics")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_agent()
```

---

## Step 6: Run the Test

Execute the script:

```bash
python langsmith/test_agent.py
```

### Expected Output:

```
✅ LangSmith configuration detected
   Project: my-first-agent
   Tracing: true

======================================================================
Testing ReAct Agent with LangSmith Tracing
======================================================================

──────────────────────────────────────────────────────────────────────
Test 1: What is the weather in Philadelphia?
──────────────────────────────────────────────────────────────────────

✓ Response: The weather in Philadelphia is: Sunny and 72°F with light winds

──────────────────────────────────────────────────────────────────────
Test 2: What is the capital of France?
──────────────────────────────────────────────────────────────────────

✓ Response: The capital of France is Paris.

──────────────────────────────────────────────────────────────────────
Test 3: Can you check the weather in Seattle and San Francisco?
──────────────────────────────────────────────────────────────────────

✓ Response: Here's the weather for both cities:
- Seattle: Rainy as always, 55°F
- San Francisco: Foggy and cool, 58°F

======================================================================
Testing Complete!
======================================================================

📊 View traces in LangSmith:
   → https://smith.langchain.com/
   → Project: my-first-agent
```

---

## Step 7: Verify in LangSmith Dashboard

1. Open your browser and go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project (e.g., "my-first-agent")
3. You should see **3 traces** corresponding to the 3 test queries

### What to Look For:

In each trace, you'll see:

- **📋 Input/Output**: The full query and final response
- **🔧 Tool Calls**: When `get_weather` was invoked (for queries 1 and 3)
- **🤖 LLM Calls**: All interactions with GPT-4o-mini
- **⏱️ Latency**: How long each step took
- **💰 Token Usage**: Input/output tokens and estimated cost
- **🔗 Execution Flow**: Visual graph showing the workflow path

### Example Trace Structure:

```
Query: "What is the weather in Philadelphia?"
  └─ Agent Node
      └─ LLM Call (decides to use tool)
          └─ Tool Call: get_weather(city="Philadelphia")
              └─ Tool Output: "Sunny and 72°F..."
                  └─ LLM Call (formats final response)
                      └─ Final Answer
```

---

## Key Concepts

### 1. **Automatic Tracing**
Once `LANGSMITH_TRACING=true` is set, **all** LangChain operations are automatically traced—no code changes needed!

### 2. **Tool Calling vs. LLM Responses**
- Query 1 & 3: Agent **uses the tool** because they ask about weather
- Query 2: Agent **answers directly** using LLM knowledge (no tool needed)

### 3. **LangGraph vs. LCEL**
This example uses **LangGraph** (the v1.0 standard) instead of deprecated LCEL:
- ✅ More explicit and maintainable
- ✅ Better for complex workflows with cycles
- ✅ Built-in state management and persistence
- ❌ LCEL (pipe operators like `|`) is **removed** in v1.0

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"
**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### Issue: "No traces appear in LangSmith"
**Solution**: Verify all environment variables are set:
```bash
echo $LANGSMITH_TRACING  # Should be "true"
echo $LANGSMITH_API_KEY  # Should show your key
```

### Issue: "401 Unauthorized" error
**Solution**: Your API key may be invalid. Create a new one in LangSmith settings.

### Issue: Import errors
**Solution**: Ensure you have the correct package versions:
```bash
pip install --upgrade langchain langchain-openai langsmith langgraph
```

---

## Next Steps

Now that LangSmith is working, you can:

1. **Build more complex agents**: Add database tools, search capabilities, or custom APIs
2. **Set up evaluations**: Create test datasets to benchmark your agent's performance
3. **Monitor production**: Track real user interactions and identify issues
4. **A/B testing**: Compare different prompts or models to optimize performance

### Useful Resources:
- [LangSmith Documentation](https://docs.langchain.com/langsmith)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph)
- [LangChain v1.0 Migration Guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)

---

## Summary Checklist

- ✅ Created LangSmith API key
- ✅ Installed dependencies (`langchain`, `langchain-openai`, `langsmith`, `langgraph`)
- ✅ Configured environment variables
- ✅ Built a ReAct-style agent using LangGraph
- ✅ Tested with multiple query types
- ✅ Verified traces appear in LangSmith dashboard
- ✅ Understood the difference between tool calls and direct LLM responses

**Congratulations!** You've successfully integrated LangSmith with LangChain v1.0. 🎉
