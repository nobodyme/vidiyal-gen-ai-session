import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Make sure to set this in your environment
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.  "
          "Please set it in your .env file or directly in the environment.")
    exit()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Use the model you prefer
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

def summarize_pdf(pdf_path):
    """
    Summarize a PDF document using LangChain's map-reduce approach.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: The final summary
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Map prompt - for summarizing individual chunks
    map_template = "Write a concise summary of the following: {docs}."
    map_prompt = ChatPromptTemplate([("human", map_template)])
    map_chain = LLMChain(llm=llm, prompt=map_prompt)


    # Reduce
    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)


    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=1000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    result = map_reduce_chain.invoke(split_docs)
    print(result["output_text"])


# To remove deprecation warnings you can move to langgraph summarization version described here - https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/
if __name__ == "__main__":
    # Example usage
    pdf_file_path = "./sample.pdf"
    summary = summarize_pdf(pdf_file_path)
    print(summary)