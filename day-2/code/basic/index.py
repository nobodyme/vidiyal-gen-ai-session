from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyDuCT6Qr5SYUso1QbrWnheJhQ97rF6vUR4")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="A school fundraiser sells tickets at $5 for children and $8 for adults. By the end of the day, they sold 230 tickets total and collected $1,490 in revenue. How many adult tickets were sold?",
    config=types.GenerateContentConfig(
        temperature=0
    ),
)

print(response.text)