import sys
import os
from pydantic import BaseModel

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.llm.llm_factory import LLMFactory
from dotenv import load_dotenv

# Load environment variables explicitly
load_dotenv(verbose=True)

# Define a Pydantic model for structured output test
class Movie(BaseModel):
    title: str
    year: int
    director: str

def test_gemini_client():
    print("Testing Gemini Client...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"GEMINI_API_KEY present: {'GEMINI_API_KEY' in os.environ}")

    
    # Configuration for Gemini
    config = {
        "client": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    try:
        client = LLMFactory.create_client(config)
        print("Client created successfully.")
    except Exception as e:
        print(f"Failed to create client: {e}")
        return

    # Test 1: Normal Chat
    print("\n--- Test 1: Normal Chat ---")
    prompt = "Tell me a short joke about programming."
    try:
        response = client.generate(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test 2: Structured Output
    print("\n--- Test 2: Structured Output ---")
    prompt = "Give me information about the movie 'Inception'."
    try:
        response = client.generate(prompt, format=Movie)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        if isinstance(response, dict) and "title" in response:
             print("Structured output verification: PASSED")
        else:
             print("Structured output verification: FAILED")
    except Exception as e:
        print(f"Test 2 failed: {e}")

    # Test 3: Grounding
    print("\n--- Test 3: Grounding ---")
    prompt = "What is the latest version of Python as of today?"
    try:
        response = client.generate(prompt, grounding=True)
        print(f"Prompt: {prompt}")
        print(f"Response (Grounded): {response}")
    except Exception as e:
        print(f"Test 3 failed: {e}")

if __name__ == "__main__":
    test_gemini_client()
