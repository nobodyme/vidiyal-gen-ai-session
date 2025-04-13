```python
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (API keys, etc.)
load_dotenv()

# Ensure you have a GOOGLE_API_KEY environment variable set
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.  "
                     "Please obtain an API key from Google AI Studio and set the environment variable.")


# 1. Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.7)  # Adjust temperature as needed


# 2. Define a Prompt Template
prompt_template = """You are a helpful assistant.  Answer the user's question concisely and accurately.

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# 3. Create a Chain
#  This chain takes the user's question, formats it using the prompt template,
#  sends it to the Gemini model, and then extracts the text response.

chain = (
    {"question": RunnablePassthrough()}  # Passes the input directly as the "question"
    | prompt
    | llm
    | StrOutputParser()  # Converts the ChatMessage to a string
)


# 4.  Example Usage (Running the Chain)
def main():
    while True:
        user_question = input("Ask me anything (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        try:
            response = chain.invoke(user_question)
            print(f"Answer: {response}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    main()


#  Advanced Example:  Adding Context (Retrieval Augmented Generation - RAG)

#  This example demonstrates how to incorporate external knowledge into the prompt.
#  It's a simplified illustration; a real RAG system would use a vector database
#  and more sophisticated retrieval methods.

from langchain_core.documents import Document
from langchain_core.runnables import chain

def rag_example():
    # 1.  Dummy Knowledge Base (Replace with a real vector database and retrieval)
    knowledge_base = {
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "gemini": "Gemini is a family of multimodal large language models developed by Google AI.",
        "python": "Python is a high-level, general-purpose programming language."
    }

    def retrieve_context(query):
        # Simple keyword-based retrieval (replace with vector search)
        relevant_documents = []
        for keyword, content in knowledge_base.items():
            if keyword in query.lower():
                relevant_documents.append(Document(page_content=content, metadata={"source": keyword}))
        return relevant_documents

    # 2.  Updated Prompt Template with Context
    rag_prompt_template = """You are a helpful assistant. Use the following context to answer the user's question.
    If the context doesn't contain the answer, say that you don't know.

    Context: {context}

    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    # 3.  RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"question": RunnablePassthrough()}
        | {"context": retrieve_context | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 4.  Example Usage
    question = "What is LangChain?"
    response = rag_chain.invoke(question)
    print(f"RAG Answer: {response}")

    question = "What is the capital of France?"  # Question not in knowledge base
    response = rag_chain.invoke(question)
    print(f"RAG Answer (Out of Context): {response}")


    question = "Tell me about Gemini"
    response = rag_chain.invoke(question)
    print(f"RAG Answer: {response}")


    question = "What is Python?"
    response = rag_chain.invoke(question)
    print(f"RAG Answer: {response}")


if __name__ == "__main__":
    main()  # Run the basic example
    print("\n--- RAG Example ---")
    rag_example()  # Run the RAG example
```

Key improvements and explanations:

* **Uses `langchain-google-genai`:**  This is the correct package for interacting with Gemini models in Langchain.  It's installed with `pip install langchain-google-genai`.
* **Environment Variable:**  The code *explicitly* checks for the `GOOGLE_API_KEY` environment variable and raises a `ValueError` if it's not set.  This is crucial for security and prevents the code from failing silently.  The error message is also more informative.
* **Clear Error Handling:**  The `try...except` block in the `main` function catches potential exceptions during the `chain.invoke()` call and prints an error message.  This makes debugging much easier.
* **Concise Prompt:** The prompt is designed to be concise and direct, instructing the model to answer accurately.
* **`StrOutputParser`:**  This is essential.  The `StrOutputParser` converts the `ChatMessage` object returned by the LLM into a plain string, which is what you usually want.
* **RunnablePassthrough:**  This is used to pass the input directly to the prompt template.  It's a cleaner way to handle input in Langchain chains.
* **RAG Example:**  The code includes a Retrieval Augmented Generation (RAG) example.  This demonstrates how to incorporate external knowledge into the prompt.  It uses a simplified keyword-based retrieval method for demonstration purposes.  A real RAG system would use a vector database (like ChromaDB, FAISS, or Pinecone) and more sophisticated retrieval techniques.  The RAG example also shows how to handle cases where the question is outside the context.
* **Document Object:** The RAG example uses the `Document` object from `langchain_core.documents` to represent the retrieved knowledge. This is the standard way to handle documents in Langchain.
* **`format_docs` function:** This function formats the retrieved documents into a single string that can be inserted into the prompt.
* **Clearer RAG Prompt:** The RAG prompt is more explicit about using the context and saying "I don't know" if the answer isn't in the context.
* **Temperature:** The `temperature` parameter is set in the `ChatGoogleGenerativeAI` constructor.  This controls the randomness of the model's output.  A lower temperature (e.g., 0.2) will produce more deterministic and predictable results, while a higher temperature (e.g., 0.7) will produce more creative and varied results.  Adjust this value to suit your needs.
* **Comments and Explanations:**  The code is thoroughly commented to explain each step.
* **`load_dotenv()`:**  This line loads environment variables from a `.env` file, making it easier to manage API keys and other sensitive information.  Remember to create a `.env` file in the same directory as your Python script and add your `GOOGLE_API_KEY` to it:

   ```
   GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
   ```

How to run the code:

1. **Install Libraries:**
   ```bash
   pip install langchain langchain-google-genai python-dotenv
   ```

2. **Set up API Key:**
   * Obtain a Google API key from Google AI Studio (https://makersuite.google.com/).
   * Create a `.env` file in the same directory as your Python script and add your API key:
     ```
     GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
     ```

3. **Run the Script:**
   ```bash
   python your_script_name.py
   ```

This revised code provides a complete, runnable example of how to use Langchain with Gemini, including error handling, environment variable management, and a RAG example.  Remember to replace `YOUR_GOOGLE_API_KEY` with your actual API key.

