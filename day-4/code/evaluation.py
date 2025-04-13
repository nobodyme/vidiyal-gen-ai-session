import os

from tqdm.notebook import tqdm
from datasets import load_dataset
from qdrant_client import QdrantClient
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
import deepeval
from pydantic import BaseModel
from google import genai
from google.genai import types
import instructor
from deepeval.models import DeepEvalBaseLLM


# Get a FREE forever cluster at https://cloud.qdrant.io/
# More info: https://qdrant.tech/documentation/cloud/create-cluster/
QDRANT_URL = os.environ.get("QDRANT_CLUSTER_URL")  
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")
if not QDRANT_URL or not QDRANT_API_KEY or not COLLECTION_NAME:
    print("Error: QDRANT_CLUSTER_URL, QDRANT_API_KEY, or QDRANT_COLLECTION_NAME not found in environment variables."
          "Please set them in your .env file or directly in the environment.")
    exit()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Make sure to set this in your environment
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.  "
          "Please set it in your .env file or directly in the environment.")
    exit()

EVAL_SIZE = 10
RETRIEVAL_SIZE = 3


class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.model = genai.Client(api_key=GOOGLE_API_KEY)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> BaseModel:
        client = self.load_model()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0
            ),
        )
        return response.text

    async def a_generate(self, prompt: str) -> BaseModel:
        return self.generate(prompt)

    def get_model_name(self):
        return "Gemini 2.0 Flash"

dataset = load_dataset("atitaarora/qdrant_doc", split="train")

langchain_docs = [
    LangchainDocument(
        page_content=doc["text"], metadata={"source": doc["source"]}
    )
    for doc in tqdm(dataset)
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

docs_contents, docs_metadatas = [], []

for doc in docs_processed:
    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
        docs_contents.append(doc.page_content)
        docs_metadatas.append(doc.metadata)
    else:
        print(
            "Warning: Some documents do not have 'page_content' or 'metadata' attributes."
        )

# Uses FastEmbed - https://qdrant.tech/documentation/fastembed/
# To generate embeddings for the documents
# The default model is `BAAI/bge-small-en-v1.5`
qdrant_client.add(
    collection_name=COLLECTION_NAME,
    metadata=docs_metadatas,
    documents=docs_contents,
)


def query_with_context(query, limit):

    search_result = qdrant_client.query(
        collection_name=COLLECTION_NAME, query_text=query, limit=limit
    )

    contexts = [
        "document: " + r.document + ",source: " + r.metadata["source"]
        for r in search_result
    ]
    prompt_start = """ You're assisting a user who has a question based on the documentation.
        Your goal is to provide a clear and concise response that addresses their query while referencing relevant information
        from the documentation.
        Remember to:
        Understand the user's question thoroughly.
        If the user's query is general (e.g., "hi," "good morning"),
        greet them normally and avoid using the context from the documentation.
        If the user's query is specific and related to the documentation, locate and extract the pertinent information.
        Craft a response that directly addresses the user's query and provides accurate information
        referring the relevant source and page from the 'source' field of fetched context from the documentation to support your answer.
        Use a friendly and professional tone in your response.
        If you cannot find the answer in the provided context, do not pretend to know it.
        Instead, respond with "I don't know".

        Context:\n"""

    prompt_end = f"\n\nQuestion: {query}\nAnswer:"

    prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end

    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    res = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0
        ),
    )
    print("response", res)
    return (contexts, res.text)


qdrant_qna_dataset = load_dataset("atitaarora/qdrant_doc_qna", split="train")


def create_deepeval_dataset(dataset, eval_size, retrieval_window_size):
    test_cases = []
    for i in range(eval_size):
        entry = dataset[i]
        question = entry["question"]
        answer = entry["answer"]
        context, rag_response = query_with_context(
            question, retrieval_window_size
        )
        test_case = deepeval.test_case.LLMTestCase(
            input=question,
            actual_output=rag_response,
            expected_output=answer,
            retrieval_context=context,
        )
        test_cases.append(test_case)
    return test_cases


test_cases = create_deepeval_dataset(
    qdrant_qna_dataset, EVAL_SIZE, RETRIEVAL_SIZE
)

evaluation_model=CustomGeminiFlash()
deepeval.evaluate(
    test_cases=test_cases,
    metrics=[
        deepeval.metrics.AnswerRelevancyMetric(model=evaluation_model),
        deepeval.metrics.FaithfulnessMetric(model=evaluation_model),
        deepeval.metrics.ContextualPrecisionMetric(model=evaluation_model),
        deepeval.metrics.ContextualRecallMetric(model=evaluation_model),
        deepeval.metrics.ContextualRelevancyMetric(model=evaluation_model),
    ],
)