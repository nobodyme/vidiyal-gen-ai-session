```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# Load environment variables (API keys, etc.)
load_dotenv()

# --- 1.  Setup Google Gemini API Key ---
# Ensure you have your GOOGLE_API_KEY in your .env file
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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
# Choose your model (gemini-1.5-pro, gemini-pro, etc.).  Adjust temperature and other parameters as desired.
llm = ChatGoogleGenerativeAI(
    model_name="gemini-1.5-pro-latest",  #  or "gemini-pro", "gemini-1.5-pro-latest"  Check Google AI Studio for available models.
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,   # Adjust for more/less randomness
    convert_system_message_to_human=True # Very important for good results.  Some LLMs treat system messages differently.
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
chain = LLMChain(llm=llm, prompt=prompt, output_parser=StringOutputParser())


# --- 6. Interact with the LLM ---
def ask_gemini(query):
    """Sends a query to Gemini and returns the response."""
    try:
        response = chain.run(query) # Use chain.run() for a simple response.  For more control, use chain.invoke().
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
```

Key improvements and explanations:

* **Clearer Error Handling:** The code now explicitly checks for the `GOOGLE_API_KEY` and provides an informative error message if it's missing.  It also includes a `try...except` block in the `ask_gemini` function to catch potential errors during the LLM interaction.  This is crucial for debugging and robustness.
* **`dotenv` Loading:**  The code uses `load_dotenv()` to load the API key from a `.env` file.  This is a best practice to avoid hardcoding sensitive information in your script.  Make sure you have the `python-dotenv` package installed (`pip install python-dotenv`).
* **Explicit Output Parser:**  I've added a basic `StringOutputParser`.  While not strictly *required* for string output, it's a good practice to include one. You can then expand it to handle JSON or other structured data if you want to get more structured responses.
* **LLM Model Selection:**  The code allows you to choose your desired Gemini model (e.g., `gemini-pro`, `gemini-1.5-pro`).  Critically, it includes  `gemini-1.5-pro-latest`.  Use the specific model you have access to and that fits your needs.  *Check Google AI Studio to confirm which models are available to you.*
* **Temperature Control:**  The `temperature` parameter is included in the `ChatGoogleGenerativeAI` initialization. Adjusting the temperature allows you to control the randomness of the model's responses. Higher values (e.g., 0.9) will result in more creative but potentially less coherent answers, while lower values (e.g., 0.2) will result in more deterministic and predictable answers.
* **Prompt Template:** Using `PromptTemplate` makes it easier to modify and manage the instructions given to the LLM.
* **LLMChain:**  Using `LLMChain` streamlines the process of combining the LLM, prompt, and output parser.
* **Interactive Loop:** The `if __name__ == "__main__":` block creates an interactive loop, allowing you to ask multiple questions to Gemini without restarting the script.
* **`convert_system_message_to_human=True`:** This is a *very important* addition.  Gemini models often treat system messages (parts of the prompt that set the overall behavior) differently than human messages. Setting this parameter ensures the system message is handled correctly, leading to better responses.
* **Clear Comments and Structure:** I've added comments to explain each step of the process.  The code is well-structured and easy to understand.
* **`chain.run()` vs. `chain.invoke()`:** The example uses `chain.run(query)` which is a simpler way to get the output of the chain.  For more advanced control (passing callbacks, configuration options, etc.) you would use `chain.invoke(query)`.  The comment explains this distinction.

How to Run the Code:

1. **Install Dependencies:**
   ```bash
   pip install langchain langchain-google-genai python-dotenv
   ```
2. **Create a `.env` file:** In the same directory as your Python script, create a file named `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
   ```
   (Replace `YOUR_GOOGLE_API_KEY` with your actual API key.)
3. **Run the Script:**
   ```bash
   python your_script_name.py  # Replace your_script_name.py with the name of your file
   ```

Now you can start interacting with Gemini through the command line!

