import time

from google import genai
from google.genai import types

from src.config import get_api_key

CANDIDATES = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite", "gemini-flash-latest",
]

client = genai.Client(
    api_key=get_api_key(),
    http_options=types.HttpOptions(
        timeout=20_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

for model in CANDIDATES:
    start = time.time()
    try:
        r = client.models.generate_content(model=model, contents="Reply with the word OK")
        print(f"OK     {model:<28} {time.time() - start:4.1f}s  {r.text.strip()[:20]!r}")
    except Exception as e:
        print(f"FAILED {model:<28} {time.time() - start:4.1f}s  {str(e)[:90]}")