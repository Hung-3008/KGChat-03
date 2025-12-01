import os
from google import genai
from google.genai import types
from typing import Optional, Dict, Union
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    def __init__(self, config: dict, **kwargs):
        self.model_name = config.get("model", "gemini-1.5-flash")
        self.temperature = config.get("temperature", 0.85)
        self.top_p = config.get("top_p", 0.9)
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        
        # Basic generation config
        self.config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _normal_response(self, prompt: str, grounding: bool = False) -> str:
        tools = None
        if grounding:
            tools = [types.Tool(google_search=types.GoogleSearch())]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                tools=tools
            )
        )
        
        return response.text

    def _structured_response(self, prompt: str, format: BaseModel) -> Dict:
        # For structured output, we pass the schema in the generation config
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                response_mime_type="application/json",
                response_schema=format
            )
        )
        
        # Parse the JSON response
        import json
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback or error handling
            return {"error": "Failed to parse JSON response", "raw_content": response.text}

    def generate(self, prompt: str, format: Optional[BaseModel] = None, grounding: bool = False) -> Union[str, Dict]:
        if format:
            return self._structured_response(prompt, format)
        else:
            return self._normal_response(prompt, grounding)
