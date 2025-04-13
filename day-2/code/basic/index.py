import os

from google import genai
from google.genai import types

# put your GOOGLE API key in your env
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.  "
          "Please set it in your .env file or directly in the environment.")
    exit()

client = genai.Client(api_key=GOOGLE_API_KEY)

# We are going to use this base model to generate the same code using langchain and put it in langchain/index.py
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Write an entire python code end to end that uses langchain to talk to gemini models",
    config=types.GenerateContentConfig(
        temperature=0
    ),
)

print(response.text)