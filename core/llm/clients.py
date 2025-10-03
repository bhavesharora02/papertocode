import os
from langchain_google_genai import ChatGoogleGenerativeAI

def make_gemini(model: str = "gemini-2.5-pro", temperature: float = 0.4):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

