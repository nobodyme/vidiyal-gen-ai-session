import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# --- 1.  Setup Google Gemini API Key ---
# Ensure you have your GOOGLE_API_KEY in your .env file
GOOGLE_API_KEY = "<YOUR GOOGLE API KEY>"

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.  "
          "Please set it in your .env file or directly in the environment.")
    exit()

# --- 2. Define Output Parser (Optional but Recommended) ---
#  This example uses a simple string output parser. You can create more sophisticated
#  parsers to handle structured output (e.g., JSON, lists) if needed.
class StringOutputParser(BaseOutputParser):
    """Parses the output of an LLM to a string."""

    def parse(self, text: str) -> str:
        """Parses the given text into a string."""
        return text

# --- 3.  Define LLM Model ---
llm = ChatGoogleGenerativeAI(
    #  Initialize the LLM with your Google API key and model name.
    #  You can also set other parameters like temperature, max tokens, etc.
    #  Refer to the Langchain documentation for more options.
    model="gemini-2.0-flash",  
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
)

# --- 4. Define Prompt Template ---
#   Create a prompt template to structure your interaction with the LLM.
#   This allows you to easily change the instructions and format of your input.
template = """
You are a helpful and concise assistant.

User: {user_input}
"""
prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

# --- 5. Create LLMChain ---
#  Combine the LLM, Prompt, and Output Parser into an LLMChain.
chain = prompt | llm | StringOutputParser

# --- 6. Interact with the LLM ---
def ask_gemini(query):
    """Sends a query to Gemini and returns the response."""
    try:
        response = chain.invoke(query)
        return response
    except Exception as e:
        print(f"Error interacting with Gemini: {e}")
        return None

# --- 7. Example Usage ---
if __name__ == "__main__":
    while True:
        user_query = input("Ask Gemini (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        gemini_response = ask_gemini(user_query)

        if gemini_response:
            print(f"Gemini: {gemini_response}")