from openai import OpenAI
from typing import Optional, Dict, Union, Any
from pydantic import BaseModel
import json

class VLLMClient:
    def __init__(self, config: dict = {}, **kwargs):
        """
        Initialize VLLM Client. 
        It uses the OpenAI-compatible API provided by vLLM.
        
        Config expects:
        - base_url: URL of the vLLM server (default: http://localhost:8000/v1)
        - model: Model name to use (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
        - temperature: Sampling temperature (default: 0.8)
        - top_p: Top-p sampling (default: 0.95)
        - max_tokens: Max tokens to generate (default: 1024)
        """
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.api_key = config.get("api_key", "EMPTY") # vLLM usually doesn't require an API key
        self.model = config.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.95)
        self.max_tokens = config.get("max_tokens", 4096) # Increased default for reasoning models
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=600.0, # Increase timeout to 10 minutes
        )

    def _normal_response(self, prompt: str) -> str:
        """
        Standard text generation.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _structured_response(self, prompt: str, format: Union[BaseModel, Dict, Any]) -> Dict:
        """
        Structured output generation using vLLM's guided decoding.
        
        Args:
            prompt: User prompt
            format: A Pydantic model class, a dict schema, or a JSON schema.
        """
        import re
        
        # Determine the schema to use
        json_schema = None
        
        if isinstance(format, dict):
             if "properties" in format or "type" in format:
                json_schema = format
             else:
                 json_schema = format
        elif hasattr(format, "model_json_schema"):
             json_schema = format.model_json_schema()
        else:
            raise ValueError("Format must be a Pydantic model or a JSON schema dict.")

        extra_body = {"guided_json": json_schema}
        
        # Add system prompt to guide the model, as recommended
        messages = [
            {
                "role": "system", 
                "content": "You are an AI assistant that extracts information in JSON format. Please output valid JSON matching the schema. Do not generate any <think> blocks or reasoning steps. Output ONLY the JSON."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Reasoning models (like Qwen) often output <think> blocks.
        # We need to strip them to find the JSON.
        # Remove <think>...</think> blocks
        content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Also remove potential markdown code fences ```json ... ```
        content_clean = re.sub(r'^```json\s*', '', content_clean, flags=re.MULTILINE)
        content_clean = re.sub(r'```$', '', content_clean, flags=re.MULTILINE).strip()

        # If empty (maybe all thought?) try original content just in case
        if not content_clean:
             content_clean = content

        try:
            return json.loads(content_clean)
        except json.JSONDecodeError:
            # Try to find the first '{' and the last '}'
            try:
                start = content_clean.index('{')
                end = content_clean.rindex('}') + 1
                return json.loads(content_clean[start:end])
            except (ValueError, json.JSONDecodeError):
                print(f"Warning: Failed to decode JSON from vLLM response: {content[:200]}...")
                return {"raw_content": content, "error": "JSONDecodeError"}

    def generate(self, prompt: str, format: Optional[Union[BaseModel, Dict]] = None) -> Union[str, Dict]:
        """
        Unified generation method.
        """
        if format:
            return self._structured_response(prompt, format)
        else:
            return self._normal_response(prompt)
