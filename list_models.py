from google import genai
from src.config import get_api_key

client = genai.Client(api_key=get_api_key())
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)