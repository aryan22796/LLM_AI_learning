import os 
from dotenv import load_dotenv
from google import genai
load_dotenv()

# Load environment variables from .env file
client =genai.Client(api_key=os.getenv("GENAI_API_KEY"))
print(os.getenv("GENAI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a short story about a robot learning to love.")

print(response.text)

