import os
from google import genai
from google.genai import types
import threading
from typing import Optional, Dict, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import re
import logging
from google.genai import errors

logger = logging.getLogger("gemini_client")

load_dotenv()


class GeminiClient:
    def __init__(self, config: dict, **kwargs):
        self.model_name = config.get("model", "gemini-2.5-flash")
        self.temperature = config.get("temperature", 0.85)
        self.top_p = config.get("top_p", 0.9)
        
        # Mode: "api_key" or "vertex"
        self.mode = config.get("mode", "api_key")
        
        # RPM-based rate limiting (used by both modes)
        self.rpm = config.get("rpm", 5)
        self.min_interval = 60.0 / self.rpm if self.rpm > 0 else 0
        self.last_request_time = 0
        self.lock = threading.Lock()
        
        if self.mode == "vertex":
            # Vertex AI mode
            self.project = config.get("project", "kgchat-481715")
            self.location = config.get("location", "us-central1")
            logger.info(f"Initializing GeminiClient in Vertex AI mode (project={self.project}, location={self.location}, rpm={self.rpm})")
            
            self.client = genai.Client(
                vertexai=True, 
                project=self.project, 
                location=self.location
            )
            
            self.key_manager = None
            self.api_key = None
            
        else:
            # API Key mode
            self.key_manager = config.get("key_manager")
            if self.key_manager:
                self.api_key = self.key_manager.get_key()
                if not self.api_key:
                     raise ValueError("KeyManager returned No Available Keys!")
            else:
                self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
            
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables or config")

            masked_key = self.api_key[:5] + "..." + self.api_key[-3:] if self.api_key else "None"
            logger.info(f"Initializing GeminiClient in API key mode with key: {masked_key}, rpm={self.rpm}")

            self.client = genai.Client(api_key=self.api_key)
        
        # Basic generation config
        self.config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _wait_for_rate_limit(self):
        # RPM-based limiting for both modes
        if self.min_interval <= 0:
            return
            
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def _rotate_key(self):
        # Key rotation only works in API key mode
        if self.mode == "vertex":
            return False
            
        if not self.key_manager:
            return False
        
        new_key = self.key_manager.get_key(current_key=self.api_key)
        if new_key:
            self.api_key = new_key
            self.client = genai.Client(api_key=self.api_key)
            # Reset rate limiter or keep shared?
            # Ideally reset last_request_time since it's a new key
            with self.lock:
                self.last_request_time = 0
            return True
        return False

    def generate(self, prompt: str, format: Optional[BaseModel] = None, grounding: bool = False) -> Union[str, Dict]:
        max_retries = 5
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                
                if format:
                    config.response_mime_type = "application/json"
                    config.response_schema = format
                    
                if grounding:
                    config.tools = [types.Tool(google_search=types.GoogleSearch())]

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                
                if format:
                    import json
                    try:
                        return json.loads(response.text)
                    except:
                        return {"error": "JSON Parse Error", "raw": response.text}
                return response.text if response.text else ""

            except Exception as e:
                # Retry logic for rate limits/overload
                err_str = str(e)
                if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str or "SSL" in err_str or "Connection" in err_str:
                     # Try rotation first!
                     if self.key_manager and ("429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str):
                         logger.warning(f"Hit limit ({e.code if hasattr(e, 'code') else '429'}). Rotating key...")
                         if self._rotate_key():
                             logger.info(f"Rotated to new key. Retrying immediately.")
                             continue
                         else:
                             logger.error("Rotation failed: No keys available. Sleeping...")
                     
                     # Check if we should wait
                     wait_time = base_delay * (2 ** attempt)
                     match = re.search(r"Please retry in (\d+(\.\d+)?)s", err_str)
                     if match:
                         wait_time = float(match.group(1)) + 1.0
                     
                     if attempt < max_retries - 1:
                         time.sleep(wait_time)
                         continue
                
                # Fatal error
                logger.error(f"Generate failed: {e}")
                return {"error": str(e)} if format else ""
        return ""
