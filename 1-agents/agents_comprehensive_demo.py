"""
LangChain Agents - Comprehensive Demo
======================================

This demo covers all three core components of LangChain Agents:
1. Models (Static and Dynamic)
2. Tools (Basic tools and Error Handling)
3. System Prompts (Static and Dynamic)

An agent is like a smart assistant that uses a language model as its "brain" 
to reason through tasks. It can use tools (functions) to gather information 
and take actions, working in a loop until it finds the answer.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict


# =============================================================================
# CORE COMPONENT 2: TOOLS
# =============================================================================
# Tools give your agent superpowers! They're Python functions the agent can 
# call to get information or perform actions.

@tool("search_recipe")
def find_recipe(dish_name: str) -> str:
    """
    Searches for a recipe for a specific dish.
    
    This tool simulates searching a recipe database. In a real application,
    this would query an actual database or recipe API.
    
    Args:
        dish_name (str): The name of the dish to search for.
    
    Returns:
        str: A message indicating that a recipe has been found.
    """
    # In a real app, this would query a database or API
    return f"Found a classic recipe for {dish_name}."


@tool
def check_pantry(ingredient: str) -> str:
    """
    Checks if an ingredient is available in the pantry.
    
    This tool checks a simulated pantry inventory. In production, this could
    connect to a smart fridge API or inventory management system.
    
    Args:
        ingredient (str): The ingredient to check for.
    
    Returns:
        str: A message indicating whether the ingredient is available.
    """
    # Simulated pantry inventory
    pantry = ["flour", "sugar", "eggs", "butter", "milk"]
    
    if ingredient.lower() in pantry:
        return f"Yes, you have {ingredient} in the pantry."
    return f"Sorry, you don't have {ingredient}."


# =============================================================================
# CORE COMPONENT 1: THE MODEL (STATIC APPROACH)
# =============================================================================

def demo_static_model_simple():
    """
    Demonstrates the simplest way to create an agent with a static model.
    
    A static model means the agent uses the same AI model for all requests.
    This is the most common approach and works well for most use cases.
    
    Here we use a simple string to specify the model - LangChain automatically
    figures out which provider (OpenAI, Anthropic, etc.) to use.
    """
    print("\n" + "="*80)
    print("DEMO 1: Static Model (Simple String)")
    print("="*80)
    
    # Create an agent with a simple model string
    # The "openai:gpt-4o" format tells LangChain to use OpenAI's GPT-4o model
    agent = create_agent(
        "openai:gpt-4o",
        tools=[find_recipe, check_pantry]
    )
    
    # Invoke the agent with a user query
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Do I have flour? If yes, find me a pancake recipe."}]
    })
    
    print("\nAgent Response:")
    print(result)
    print("\n[+] This approach is simple and works great for most cases!")


def demo_static_model_customized():
    """
    Demonstrates creating an agent with a customized static model.
    
    By creating a model instance directly, we can control parameters like:
    - temperature: Controls randomness (0=predictable, 1=creative)
    - max_tokens: Limits the response length
    - timeout: Sets a time limit for API calls
    """
    print("\n" + "="*80)
    print("DEMO 2: Static Model (Customized)")
    print("="*80)
    
    # Create a customized model instance with specific settings
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,  # Low temperature = more predictable, focused responses
        max_tokens=1500,  # Limit response length
        timeout=30        # 30 second timeout for API calls
    )
    
    # Pass the customized model instance to the agent
    agent = create_agent(
        model, 
        tools=[find_recipe, check_pantry]
    )
    
    # Invoke with a query
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Check if I have sugar and eggs for baking."}]
    })
    
    print("\nAgent Response:")
    print(result)
    print("\n[+] Custom settings give you fine-grained control over model behavior!")


# =============================================================================
# CORE COMPONENT 3: SYSTEM PROMPTS
# =============================================================================

def demo_static_system_prompt():
    """
    Demonstrates using a static system prompt.
    
    A system prompt is like an instruction manual for your agent. It defines:
    - The agent's personality and tone
    - Its goals and objectives
    - Rules it should follow
    
    Static prompts stay the same for all interactions.
    """
    print("\n" + "="*80)
    print("DEMO 3: Static System Prompt")
    print("="*80)
    
    # Create an agent with a specific personality and instructions
    agent = create_agent(
        "openai:gpt-4o",
        tools=[find_recipe, check_pantry],
        system_prompt="You are a helpful culinary assistant. Always provide clear, step-by-step instructions for recipes."
    )
    
    # The agent will now follow the system prompt's instructions
    result = agent.invoke({
        "messages": [{"role": "user", "content": "I want to make cookies."}]
    })
    
    print("\nAgent Response:")
    print(result)
    print("\n[+] System prompts shape how your agent communicates and behaves!")


# =============================================================================
# PUTTING IT ALL TOGETHER: COMPLETE AGENT EXAMPLE
# =============================================================================

def demo_complete_agent():
    """
    Demonstrates a complete agent combining all core components:
    - Customized model for controlled responses
    - Multiple tools for different capabilities
    - Clear system prompt for personality
    
    This is a production-ready pattern you can use in real applications.
    """
    print("\n" + "="*80)
    print("DEMO 4: Complete Agent (All Components Together)")
    print("="*80)
    
    # Step 1: Create a customized model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,  # Balanced between creative and predictable
        max_tokens=2000
    )
    
    # Step 2: Define our tools (already defined above)
    tools = [find_recipe, check_pantry]
    
    # Step 3: Create the agent with a clear system prompt
    agent = create_agent(
        model,
        tools=tools,
        system_prompt="""You are a friendly and knowledgeable culinary assistant. 
        Your goal is to help users find recipes and check their pantry inventory.
        Always be encouraging and provide helpful cooking tips when relevant.
        Keep your responses concise but informative."""
    )
    
    # Step 4: Invoke the agent with a real-world query
    result = agent.invoke({
        "messages": [{"role": "user", "content": "I want to bake something. What ingredients do I have that would work for baking?"}]
    })
    
    print("\nAgent Response:")
    print(result)
    print("\n[+] This is a complete, production-ready agent pattern!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
================================================================================
                                                                            
                   LANGCHAIN AGENTS - COMPREHENSIVE DEMO                    
                                                                            
  This demo walks through the three core components of LangChain agents:    
  1. Models (the "brain" that reasons through tasks)                        
  2. Tools (functions the agent can call to take actions)                   
  3. System Prompts (instructions that guide agent behavior)                
                                                                            
================================================================================
    """)
    
    print("\n[!] IMPORTANT NOTE:")
    print("This demo requires:")
    print("  - langchain")
    print("  - langchain-openai")
    print("  - An OpenAI API key set in your environment variables")
    print("\nMake sure you have these installed and configured before running!")
    
    # Uncomment the demos you want to run:
    # Note: These are commented out to prevent accidental API calls
    
    demo_static_model_simple()
    demo_static_model_customized()
    demo_static_system_prompt()
    demo_complete_agent()
    
    print("\n" + "="*80)
    print("[*] DEMO COMPLETED!")
    print("="*80)
    print("\nAll four demos have been executed successfully!")
    print("\n[+] All demos are ready to use!")

