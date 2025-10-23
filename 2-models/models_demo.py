"""
LangChain Models - Comprehensive Demo

This demo covers all the key concepts of working with Large Language Models (LLMs) in LangChain:
1. Initializing models (two different approaches)
2. Using invoke(), stream(), and batch() methods
3. Tool calling (letting the model use external functions)
4. Structured outputs (forcing responses to follow a specific format)

Think of a model as the "brain" of your AI application - it's what makes decisions,
generates text, and can even decide when to use tools (like calling APIs or databases).
"""

import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from pydantic import BaseModel, Field


# =============================================================================
# SECTION 1: INITIALIZING A MODEL
# =============================================================================
# There are two main ways to create a model instance in LangChain.
# Both approaches work the same way once initialized.

def demo_initialization():
    """
    Demonstrates two ways to initialize a model.
    
    Think of initialization as "setting up your AI brain" before you start asking it questions.
    You can use a simple string format or a class instance for more control.
    """
    print("\n" + "="*80)
    print("DEMO 1: Model Initialization")
    print("="*80)
    
    # Option 1: Quick initialization using a string
    # This is the simplest way - just specify the provider and model name
    print("\n[Option 1] Initializing with string format...")
    model_from_string = init_chat_model("openai:gpt-4o")
    print(f"[OK] Model initialized: {type(model_from_string).__name__}")
    
    # Option 2: Class-based initialization with configuration options
    # This gives you more control over the model's behavior
    print("\n[Option 2] Initializing with class instance and parameters...")
    model_from_class = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,      # Lower = more focused/deterministic responses (good for facts)
        timeout=60,           # Maximum wait time in seconds
        max_tokens=2000,      # Limit response length to control costs
    )
    print(f"[OK] Model initialized: {type(model_from_class).__name__}")
    print(f"  - Temperature: 0.1 (focused, factual responses)")
    print(f"  - Timeout: 60 seconds")
    print(f"  - Max tokens: 2000")
    
    # Let's use the configured model for a simple test
    print("\n[Testing] Asking model to summarize a financial topic...")
    response = model_from_class.invoke("In one sentence, what is a quarterly earnings report?")
    print(f"Response: {response.content}")
    
    return model_from_class


# =============================================================================
# SECTION 2: INVOKE METHOD - Single, Complete Requests
# =============================================================================
# invoke() is like asking a question and waiting for the full answer at once.

def demo_invoke(model):
    """
    Demonstrates the invoke() method for single requests.
    
    Use invoke() when you want the complete response all at once, 
    like asking a colleague a question and waiting for their full answer.
    """
    print("\n" + "="*80)
    print("DEMO 2: Invoke Method (Single Request)")
    print("="*80)
    
    # Example 1: Simple string input
    print("\n[Example 1] Simple string prompt...")
    response = model.invoke("What is the purpose of a balance sheet?")
    print(f"Response: {response.content}")
    
    # Example 2: Structured conversation with system and user messages
    # SystemMessage = Sets the AI's personality and behavior
    # HumanMessage = The actual question or request from the user
    print("\n[Example 2] Structured conversation with context...")
    conversation = [
        SystemMessage("You are a financial analyst assistant. Your responses should be formal and data-driven."),
        HumanMessage("What were the main drivers of revenue growth for ACME Corp in the last quarter?")
    ]
    response = model.invoke(conversation)
    print(f"Response: {response.content}")


# =============================================================================
# SECTION 3: STREAM METHOD - Real-time Response Generation
# =============================================================================
# stream() shows the response as it's being generated, word by word.

def demo_stream(model):
    """
    Demonstrates the stream() method for real-time responses.
    
    Use stream() when you want to see the response as it's generated,
    like watching someone type their answer in real-time.
    This provides a better user experience for long responses.
    """
    print("\n" + "="*80)
    print("DEMO 3: Stream Method (Real-time Generation)")
    print("="*80)
    
    print("\n[Streaming] Generating analysis in real-time...")
    print("Response: ", end="", flush=True)
    
    # Each "chunk" is a small piece of the response as it's generated
    # We print each chunk immediately to show progress
    for chunk in model.stream("Provide a brief analysis of the current tech sector trends."):
        print(chunk.content, end="", flush=True)
    
    print("\n")  # New line after streaming is complete


# =============================================================================
# SECTION 4: BATCH METHOD - Multiple Requests in Parallel
# =============================================================================
# batch() processes multiple requests at once, which is much faster than doing them one by one.

def demo_batch(model):
    """
    Demonstrates the batch() method for parallel processing.
    
    Use batch() when you have multiple independent requests to process.
    Think of it like asking three colleagues three different questions simultaneously
    instead of waiting for each person to answer before asking the next.
    This is very efficient for analyzing multiple stocks, documents, etc.
    """
    print("\n" + "="*80)
    print("DEMO 4: Batch Method (Parallel Processing)")
    print("="*80)
    
    # Create a list of independent financial analysis prompts
    print("\n[Batch Processing] Analyzing multiple companies at once...")
    prompts = [
        "In one sentence, summarize the latest earnings for Apple (AAPL).",
        "In one sentence, what are key risks for Google (GOOG)?",
        "In one sentence, provide a competitive analysis for Microsoft (MSFT)."
    ]
    
    print(f"Processing {len(prompts)} prompts in parallel...\n")
    
    # batch() sends all prompts at once and waits for all responses
    responses = model.batch(prompts)
    
    # Display each response
    for i, response in enumerate(responses, 1):
        print(f"[Response {i}]")
        print(response.content)
        print("-" * 40)


# =============================================================================
# SECTION 5: TOOL CALLING - Letting the Model Use External Functions
# =============================================================================
# Tool calling allows the model to interact with external systems and APIs.

def demo_tool_calling(model):
    """
    Demonstrates tool calling - allowing the model to use external functions.
    
    Tool calling is like giving the model a set of "special abilities" it can use.
    For example, instead of guessing a stock price, it can call a function to get the real price.
    The model decides WHEN and HOW to use these tools based on the user's question.
    """
    print("\n" + "="*80)
    print("DEMO 5: Tool Calling (External Functions)")
    print("="*80)
    
    # Define Tool 1: Get stock price
    # The @tool decorator tells LangChain this function can be used by the model
    # The docstring is IMPORTANT - it tells the model what this tool does
    @tool
    def get_stock_price(ticker: str) -> float:
        """Gets the current stock price for a given ticker symbol."""
        # This is a mock function - in real life, this would call a financial API
        stock_prices = {
            "ACME": 150.75,
            "AAPL": 178.50,
            "GOOG": 142.30,
            "MSFT": 380.20
        }
        return stock_prices.get(ticker.upper(), 0.0)
    
    # Define Tool 2: Get financial news
    @tool
    def get_financial_news(company_name: str) -> str:
        """Searches for the latest financial news about a company."""
        # Mock function - in reality, this would query a news API
        return f"Breaking News: {company_name} announces record profits and expansion plans."
    
    # Bind the tools to the model
    # "Binding" means attaching the tools so the model knows they exist and can use them
    print("\n[Setup] Binding tools to the model...")
    model_with_tools = model.bind_tools([get_stock_price, get_financial_news])
    print(f"[OK] Bound {len([get_stock_price, get_financial_news])} tools to the model")
    print(f"  - get_stock_price: Gets current stock prices")
    print(f"  - get_financial_news: Searches for company news")
    
    # Ask the model a question that requires using a tool
    print("\n[Question] What is the current stock price for ACME?")
    response = model_with_tools.invoke("What is the current stock price for ACME?")
    
    # The model doesn't actually execute the function - it just requests it
    # In a full agent, the agent would see this request, execute the function, 
    # and pass the result back to the model
    print("\n[Tool Calls Requested]")
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"  Tool: {tool_call['name']}")
            print(f"  Arguments: {tool_call['args']}")
            print(f"  Call ID: {tool_call['id']}")
            
            # Let's manually execute the tool to show what would happen
            if tool_call['name'] == 'get_stock_price':
                result = get_stock_price.invoke(tool_call['args'])
                print(f"  -> Result: ${result}")
    else:
        print("  No tool calls were made (model responded directly)")
        print(f"  Response: {response.content}")


# =============================================================================
# SECTION 6: STRUCTURED OUTPUTS - Forcing a Specific Response Format
# =============================================================================
# Structured outputs ensure the model's response follows a specific schema (like JSON).

def demo_structured_output(model):
    """
    Demonstrates structured outputs using Pydantic models.
    
    Structured outputs are like giving the model a form to fill out.
    Instead of getting free-form text, you get data in a specific format
    that your code can easily process and validate.
    
    This is CRITICAL for building reliable applications because you know
    exactly what fields you'll get back and what type of data they contain.
    """
    print("\n" + "="*80)
    print("DEMO 6: Structured Outputs (Predefined Schema)")
    print("="*80)
    
    # Define a Pydantic model for the output structure
    # Pydantic is like a blueprint that defines what fields we want and their types
    class FinancialSummary(BaseModel):
        """A structured summary of a company's financial health."""
        company_name: str = Field(description="The name of the company")
        ticker_symbol: str = Field(description="The stock ticker symbol")
        market_sentiment: str = Field(description="Current market sentiment, e.g., 'Bullish', 'Bearish', or 'Neutral'")
        key_takeaway: str = Field(description="A one-sentence summary of the financial outlook")
    
    print("\n[Setup] Defining output structure...")
    print("Expected fields:")
    print("  - company_name (string)")
    print("  - ticker_symbol (string)")
    print("  - market_sentiment (string)")
    print("  - key_takeaway (string)")
    
    # Attach the structure to the model
    # This tells the model: "Your response MUST match this exact format"
    structured_model = model.with_structured_output(FinancialSummary)
    
    # Provide information for the model to analyze and structure
    print("\n[Request] Asking model to analyze a report and return structured data...")
    prompt = """
    Analyze the following report and provide a summary:
    
    ACME Corp (ticker: ACME) just released strong Q3 earnings, beating analyst 
    expectations by 15%. Revenue grew 23% year-over-year. The company announced 
    plans to expand into Asian markets. Market outlook is very positive.
    """
    
    response = structured_model.invoke(prompt)
    
    # The response is now a Python object with guaranteed fields!
    print("\n[Structured Response]")
    print(f"  Company Name: {response.company_name}")
    print(f"  Ticker Symbol: {response.ticker_symbol}")
    print(f"  Market Sentiment: {response.market_sentiment}")
    print(f"  Key Takeaway: {response.key_takeaway}")
    print(f"\n  Type: {type(response)}")
    print(f"  [OK] Data is structured and validated!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
# Run all demonstrations in sequence

def main():
    """
    Main function to run all model demonstrations.
    
    This will walk through each concept step by step, showing you how to:
    1. Initialize models
    2. Use different invocation methods
    3. Enable tool calling
    4. Get structured outputs
    """
    print("\n")
    print("=" * 80)
    print(" " * 20 + "LANGCHAIN MODELS - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Initialize the model (will be used in all subsequent demos)
    model = demo_initialization()
    
    # Run each demonstration
    demo_invoke(model)
    demo_stream(model)
    demo_batch(model)
    demo_tool_calling(model)
    demo_structured_output(model)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
You've now seen all the core ways to work with models in LangChain:

1. [OK] Initialization: Setting up your model with the right configuration
2. [OK] Invoke: Getting complete responses for single requests
3. [OK] Stream: Receiving responses in real-time as they're generated
4. [OK] Batch: Processing multiple requests in parallel efficiently
5. [OK] Tool Calling: Giving the model access to external functions
6. [OK] Structured Outputs: Getting responses in a guaranteed format

These are the building blocks for creating intelligent agents and applications!
    """)


if __name__ == "__main__":
    # Check for API key before running
    # Note: We don't print or inspect the key for security reasons
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[WARNING] OPENAI_API_KEY environment variable not found!")
        print("Please set your API key before running this demo.")
        print("\nOn Windows (PowerShell):")
        print('  $env:OPENAI_API_KEY="your-api-key-here"')
        print("\nOn Windows (Command Prompt):")
        print('  set OPENAI_API_KEY=your-api-key-here')
        print("\nOn Linux/Mac:")
        print('  export OPENAI_API_KEY="your-api-key-here"')
    else:
        # API key is set, run the demos
        main()

