"""
Understanding LangChain Agents - Comprehensive Guide

This file demonstrates all the core concepts of LangChain Agents:
- Creating agents with different model configurations (static and dynamic)
- Defining and using tools
- Implementing error handling for tools
- Using system prompts (static and dynamic)
- Invoking agents with context

An agent is like a smart assistant that uses a language model (its "brain") to 
reason through tasks. It can use tools to gather information and take actions,
working in a loop: think -> use tool -> observe result -> think again.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, dynamic_prompt, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from typing import TypedDict

# ============================================================================
# SECTION 1: DEFINING TOOLS
# ============================================================================
# Tools give your agent "superpowers" - they're functions the agent can call
# to get information or perform actions. Think of them as apps on a smartphone.

@tool
def find_recipe(dish_name: str) -> str:
    """Searches for a recipe for a specific dish."""
    # In a real application, this would query a database or call an API
    # For this demo, we're simulating a simple response
    return f"Found a classic recipe for {dish_name}."

@tool
def check_pantry(ingredient: str) -> str:
    """Checks if an ingredient is available in the pantry."""
    # Simulating a pantry inventory
    pantry = ["flour", "sugar", "eggs", "butter", "milk"]
    
    if ingredient.lower() in pantry:
        return f"Yes, you have {ingredient} in the pantry."
    return f"Sorry, you don't have {ingredient}."

# ============================================================================
# SECTION 2: TOOL ERROR HANDLING
# ============================================================================
# Middleware that catches errors when tools fail and returns helpful messages
# to the model so it can try a different approach.

@wrap_tool_call
def custom_tool_error_handler(request, handler):
    """
    Handles errors during tool execution with a custom message.
    
    This prevents the agent from crashing when a tool fails. Instead,
    it receives a clear error message and can adjust its strategy.
    """
    try:
        return handler(request)
    except Exception as e:
        # Return a user-friendly error message to the model
        return ToolMessage(
            content=f"There was an issue with the tool. Please check your query. Error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

# ============================================================================
# SECTION 3: DYNAMIC MODEL SELECTION
# ============================================================================
# This middleware allows the agent to switch between models based on query complexity.
# It's like having a team where simple questions go to a junior assistant and
# complex questions go to a senior expert - optimizing both cost and performance.

# Define two different models with different capabilities
simple_model = ChatOpenAI(model="gpt-4o-mini")  # Faster, cheaper, good for simple tasks
expert_chef_model = ChatOpenAI(model="gpt-4o")  # More capable, better for complex tasks

@wrap_model_call
def dynamic_model_router(request: ModelRequest, handler) -> ModelResponse:
    """
    Selects a model based on query complexity using multiple heuristics.
    
    This function examines the user's query and decides which model to use.
    It checks for:
    1. Complex recipe names
    2. Many ingredients
    3. Advanced cooking techniques
    4. Query length
    """
    user_query = request.state["messages"][-1].content.lower()
    
    # Heuristic 1: Check for complex recipe names
    complex_recipes = ["beef wellington", "soufflé", "coq au vin", 
                       "bouillabaisse", "consommé", "croissant", "macarons"]
    has_complex_recipe = any(recipe in user_query for recipe in complex_recipes)
    
    # Heuristic 2: Count ingredients (comma-separated list indicates complexity)
    has_many_ingredients = user_query.count(',') > 4
    
    # Heuristic 3: Check for advanced cooking techniques
    advanced_techniques = ["sous vide", "flambé", "confit", "molecular", 
                          "emulsify", "temper", "clarify"]
    has_advanced_technique = any(technique in user_query for technique in advanced_techniques)
    
    # Heuristic 4: Query length (longer queries often indicate complexity)
    is_long_query = len(user_query.split()) > 15
    
    # Use expert model if ANY complexity indicator is present
    if has_complex_recipe or has_many_ingredients or has_advanced_technique or is_long_query:
        request.model = expert_chef_model
    else:
        request.model = simple_model
        
    return handler(request)

# ============================================================================
# SECTION 4: DYNAMIC SYSTEM PROMPTS
# ============================================================================
# Dynamic prompts allow the agent's behavior to adapt based on user context.
# This is like having a personal assistant that adjusts their advice based on
# your specific needs and preferences.

# Define the context schema - this tells the agent what kind of context to expect
class Context(TypedDict):
    dietary_preference: str

@dynamic_prompt
def dietary_prompt_builder(request: ModelRequest) -> str:
    """
    Generate a system prompt based on the user's dietary preference.
    
    This function builds custom instructions for the agent based on the
    user's dietary needs. The agent will only suggest recipes that match
    the specified dietary restrictions.
    """
    preference = request.runtime.context.get("dietary_preference", "none")
    base_prompt = "You are a helpful culinary assistant."
    
    if preference == "vegan":
        return f"{base_prompt} All recipes must be 100% plant-based. No animal products allowed."
    elif preference == "gluten-free":
        return f"{base_prompt} All recipes must be gluten-free. No wheat, barley, or rye."
    elif preference == "keto":
        return f"{base_prompt} All recipes must be low-carb and high-fat for a ketogenic diet."
    
    return base_prompt

# ============================================================================
# SECTION 5: CREATING AGENTS
# ============================================================================

def create_simple_agent():
    """
    Creates a basic agent with a static model and simple configuration.
    
    This is the most straightforward way to create an agent. The model is
    fixed and doesn't change during execution.
    """
    # Create a model with specific parameters
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,  # Low temperature for more predictable, consistent responses
        max_tokens=1500,  # Limit response length
        timeout=30        # Maximum time to wait for a response (seconds)
    )
    
    # Create the agent with the model and tools
    agent = create_agent(
        model, 
        tools=[find_recipe, check_pantry],
        system_prompt="You are a helpful culinary assistant. Always provide clear, step-by-step instructions for recipes."
    )
    
    return agent

def create_dynamic_agent():
    """
    Creates an advanced agent with dynamic model selection and error handling.
    
    This agent can:
    - Switch between models based on query complexity
    - Handle tool errors gracefully
    - Adapt its system prompt based on user context
    """
    agent = create_agent(
        model=simple_model,  # Start with the simple model as default
        tools=[find_recipe, check_pantry],
        middleware=[
            dynamic_model_router,       # Enables smart model switching
            custom_tool_error_handler,  # Handles tool failures gracefully
            dietary_prompt_builder      # Adapts prompt based on dietary needs
        ],
        context_schema=Context  # Defines what context the agent expects
    )
    
    return agent

# ============================================================================
# SECTION 6: INVOKING AGENTS
# ============================================================================

def demo_simple_agent():
    """
    Demonstrates how to use a simple agent with basic configuration.
    """
    print("=" * 70)
    print("DEMO 1: Simple Agent")
    print("=" * 70)
    
    agent = create_simple_agent()
    
    # Invoke the agent with a user message
    result = agent.invoke({
        "messages": [{"role": "user", "content": "I want to make lasagna."}]
    })
    
    print("\nUser: I want to make lasagna.")
    print(f"Agent: {result}")
    print()

def demo_dynamic_agent():
    """
    Demonstrates how to use a dynamic agent with context and adaptive behavior.
    """
    print("=" * 70)
    print("DEMO 2: Dynamic Agent with Vegan Context")
    print("=" * 70)
    
    agent = create_dynamic_agent()
    
    # Invoke the agent with a specific dietary context
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I want to make lasagna."}]},
        context={"dietary_preference": "vegan"}
    )
    
    print("\nUser: I want to make lasagna.")
    print("Context: Vegan dietary preference")
    print(f"Agent: {result}")
    print()

def demo_complex_query():
    """
    Demonstrates how the dynamic model router switches to a more capable model
    for complex queries.
    """
    print("=" * 70)
    print("DEMO 3: Complex Query (triggers expert model)")
    print("=" * 70)
    
    agent = create_dynamic_agent()
    
    # This query should trigger the expert model due to complex recipe name
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How do I make a perfect beef wellington with a golden, flaky crust?"}]},
        context={"dietary_preference": "none"}
    )
    
    print("\nUser: How do I make a perfect beef wellington with a golden, flaky crust?")
    print(f"Agent: {result}")
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LANGCHAIN AGENTS - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how to create and use LangChain agents with:")
    print("  • Static and dynamic models")
    print("  • Custom tools for specific tasks")
    print("  • Error handling middleware")
    print("  • Adaptive system prompts based on context")
    print()
    
    # Run the demonstrations
    # Note: These will actually call the OpenAI API if the API key is configured
    # Comment out the demos below if you want to avoid API calls
    
    print("NOTE: To run these demos, ensure your OPENAI_API_KEY environment variable is set.")
    print("If you see errors, the API key might not be configured.\n")
    
    # Uncomment these lines to run the demos:
    # demo_simple_agent()
    # demo_dynamic_agent()
    # demo_complex_query()
    
    print("All demonstrations are defined and ready to run!")
    print("Uncomment the demo function calls in the main block to execute them.")

