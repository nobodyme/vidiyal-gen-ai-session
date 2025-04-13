import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from enum import Enum
from pydantic import BaseModel
from summary import summarize_pdf

# Set your API key (in practice, use environment variables)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Make sure to set this in your environment
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.  "
          "Please set it in your .env file or directly in the environment.")
    exit()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Use the model you prefer
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,  # Lower temperature for more consistent classification
)

# Setup the output parser for structured output
intent_schema = ResponseSchema(
    name="intent",
    description="The classified intent of the user query (rag, summary, or llm)",
)

class Intents(Enum):
    RAG = "rag"
    LLM = "llm"
    SUMMARY = "summary"

class ClassifiedIntents(BaseModel):
    intent: Intents
    confidence: float
    
output_parser = PydanticOutputParser(pydantic_object=ClassifiedIntents)
format_instructions = output_parser.get_format_instructions()

# Create a prompt template for intent classification
intent_prompt_template = """
You are an intent classifier for an AI application. Your task is to classify each user query into exactly one of the following intents:

1) rag – The query requests general information from documents or knowledge base.
2) summary – The query asks for a summary or key-point distillation of a document.
3) llm – The query consists of general conversation (e.g., greetings, thanks).

Rules:
- Use the llm intent only for queries that are greetings, casual conversation, or clearly unrelated to document content.
- If the query is a general conversation but not a greeting or thanks, then default to rag.
- If the intent is not clear from the query, default to the rag intent.

Examples:
- "Hi there!" → llm
- "Can you summarize the calls from last week?" → summary
- "What are the major risks in the automotive market?" → rag
- "Thanks for your help!" → llm

User Query: {user_query}

{format_instructions}
"""

intent_prompt = PromptTemplate(
    input_variables=["user_query", "format_instructions"],
    template=intent_prompt_template,
)

def classify_intent(user_query):
    """Classifies the intent of a user query."""
    try:
        # Prepare the prompt with the user query and format instructions
        formatted_prompt = intent_prompt.format(
            user_query=user_query,
            format_instructions=format_instructions
        )
        
        # Get response from the LLM
        response = llm.invoke(formatted_prompt)

        print("Response from LLM:", response.content)
        
        # Parse the structured output
        parsed_response = output_parser.parse(response.content)
        
        # Return just the intent value
        return parsed_response.intent
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return "rag"  # Default to rag on error

def process_query(user_query):
    """Process a user query based on its intent."""
    # Classify the intent
    intent = classify_intent(user_query)
    print(f"Classified intent: {intent}")
    
    # Handle the query based on intent
    if intent == Intents.RAG:
        # return rag(query)
        return "Your question is being answered using RAG (Retrieval Augmented Generation)."
    elif intent == Intents.SUMMARY:
        return "Your question is being answered with a document summary."
    elif intent == Intents.LLM:
        return "Your question is being answered directly by the LLM."
    else:
        return "I couldn't determine how to process your query."

# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        # "Hi there!",
        "Can you summarize the latest quarterly report?",
        # "What are the major risks in the automotive market?",
        # "Thanks for your help!"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = process_query(query)
        print(f"Response: {response}")
    