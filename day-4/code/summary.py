import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.prompts import PromptTemplate

# Set your API key (in practice, use environment variables)
GOOGLE_API_KEY = "<YOUR GEMINI API KEY>"  # Replace with your actual API key

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Use the model you prefer
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,  # Lower temperature for more consistent classification
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
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Map prompt - for summarizing individual chunks
    map_prompt_template = """
    You are an expert summarizer. Your goal is to create a clear and concise summary of the following text:
    
    {text}
    
    Focus on the key points and main ideas. Be concise but comprehensive.
    
    SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    # Reduce prompt - for combining summaries into a final summary
    reduce_prompt_template = """
    You are an expert summarizer. Your goal is to create a clear and concise summary that combines the following summaries:
    
    {text}
    
    Focus on the key points and main ideas. Be concise but comprehensive.
    
    FINAL SUMMARY:
    """
    reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])
    
    # Create the map-reduce chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True
    )
    
    # Run the chain
    final_summary = summary_chain.invoke(split_docs)
    
    return final_summary["output_text"]

if __name__ == "__main__":
    # Example usage
    pdf_file_path = "./sample.pdf"
    summary = summarize_pdf(pdf_file_path)
    print(summary)