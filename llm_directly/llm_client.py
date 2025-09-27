from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
import logging
from openai import OpenAI
import ollama
import sys
load_dotenv()
logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    pass


class LLMClient(ABC):
    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass




class GeminiChat(LLMClient):
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._model = None
        self._initialize_client()


    def _initialize_client(self) -> None:
        if not self.api_key:
            raise AuthenticationError("Gemini API key is required")
        genai.configure(api_key = self.api_key)
        generation_config = {
            "temperature": 0.7,           
            # "top_p": 0.9,                 
            # "top_k": 40,                 
            "max_output_tokens": 1024,
        }
        self._model = genai.GenerativeModel(model_name = self.model_name, generation_config = generation_config)
        logger.info(f"Initialized Gemini client with model: {self.model_name}")

    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        try:
            if system_prompt:
                full_prompt = f"System:{system_prompt}\n\nUser:{user_prompt}"
            else:
                full_prompt = user_prompt
            response = self._model.generate_content(full_prompt)    
            message = response.text if response.parts else "No response generated"
            return message
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class OpenAIChat(LLMClient):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = None  
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.api_key:
            raise AuthenticationError("OpenAI API key is required")
        self._model = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI client with model: {self.model_name}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        try:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                messages = [{"role": "user", "content": user_prompt}]

            response = self._model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                # top_p=0.9
            )
            message = (
                response.choices[0].message.content
                if response and response.choices and response.choices[0].message and response.choices[0].message.content
                else "No response generated"
            )
            return message

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

class OllamaChat(LLMClient):
    def __init__(self, model_name: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        if not self.host:
            raise AuthenticationError("Ollama host is required")
        if not self.model_name:
            raise ValueError("Ollama model_name is required")
        
        self._model = ollama.Client(host=self.host)
        logger.info(f"Initialized Ollama client with host: {self.host}")
    
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = self._model.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.0,
                    "num_predict": 1024,
                    # "top_p": 0.9,
                    # "top_k": 40,
                }
            )

            if not response or "message" not in response:
                raise RuntimeError("Ollama response invalid or missing 'message'")

            message = response['message']['content'] if response.get("message") else "No response generated"
            return message

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise



class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:

        provider = provider.lower()

        if provider == "gemini":
            return GeminiChat(
                model_name=kwargs.get("model_name", "gemini-2.0-flash"),
                api_key=kwargs.get("api_key"),
            )

        elif provider == "openai":
            return OpenAIChat(
                model_name=kwargs.get("model_name", "gpt-4o-mini"),
                api_key=kwargs.get("api_key"),
            )

        elif provider == "ollama":
            return OllamaClient(
                model_name=kwargs.get("model_name", "llama3.2:3b"),
                host=kwargs.get("host", "http://localhost:11434"),
            )

        else:
            raise ValueError(f"Unknown provider: {provider}. Chọn 'gemini', 'openai', hoặc 'ollama'.")



def Operator(user_prompt: str, system_prompt: Optional[str] = None):
    print("=== LLM OPERATOR ===")
    # system_prompt = input("System prompt: ").strip()
    # user_prompt = input("User prompt: ").strip()

    print("\n=== LLM MENU ===")
    print("1) Gemini")
    print("2) OpenAI")
    print("3) Ollama")
    print("4) All model")
    choice = input("Chọn mô hình (1/2/3/4): ").strip()

    if choice == "1":
        client = GeminiChat(model_name="gemini-2.0-flash")
        ans = client.generate(user_prompt, system_prompt)
        print("Gemini:", ans)

    elif choice == "2":
        client = OpenAIChat(model_name="gpt-4o-mini")
        ans = client.generate(user_prompt, system_prompt)
        print("OpenAI:", ans)

    elif choice == "3":
        client = OllamaChat(mGeminiChaodel_name="llama3.2:3b", host="http://localhost:11434")
        ans = client.generate(user_prompt, system_prompt)
        print("Ollama:", ans)

    elif choice == "4":
        print("\nGemini:")
        client1 = GeminiChat(model_name="gemini-2.0-flash")
        print(client1.generate(user_prompt, system_prompt))

        print("\nOpenAI:")
        client2 = OpenAIChat(model_name="gpt-4o-mini")
        print(client2.generate(user_prompt, system_prompt))

        print("\nOllama:")
        client3 = OllamaChat(model_name="llama3.2:3b", host="http://localhost:11434")
        print(client3.generate(user_prompt, system_prompt))

    else:
        print("Lựa chọn không hợp lệ")
        sys.exit(1)
s = input("System prompt: ").strip()
u = input("User prompt: ").strip()    
Operator(u,s)# system_prompt = input("System prompt: ").strip()
    # user_prompt = input("User prompt: ").strip()
    # print("=== LLM OPERATOR ===")
    # system_prompt = input("System prompt: ").strip()
    # user_prompt = input("User prompt: ").strip()

    # print("\n=== LLM MENU ===")
    # print("1) Gemini")
    # print("2) OpenAI")
    # print("3) Ollama")
    # print("4) All model")
    # choice = input("Chọn mô hình (1/2/3/4): ").strip()

    # if choice == "1":
    #     client = GeminiChat(model_name="gemini-2.0-flash")
    #     ans = client.generate(user_prompt, system_prompt)
    #     print("Gemini:", ans)

    # elif choice == "2":
    #     client = OpenAIChat(model_name="gpt-4o-mini")
    #     ans = client.generate(user_prompt, system_prompt)
    #     print("OpenAI:", ans)

    # elif choice == "3":
    #     client = OllamaChat(model_name="llama3.2:3b", host="http://localhost:11434")
    #     ans = client.generate(user_prompt, system_prompt)
    #     print("Ollama:", ans)

    # elif choice == "4":
    #     print("\nGemini:")
    #     client1 = GeminiChat(model_name="gemini-2.0-flash")
    #     print(client1.generate(user_prompt, system_prompt))

    #     print("\nOpenAI:")
    #     client2 = OpenAIChat(model_name="gpt-4o-mini")
    #     print(client2.generate(user_prompt, system_prompt))

    #     print("\nOllama:")
    #     client3 = OllamaChat(model_name="llama3.2:3b", host="http://localhost:11434")
    #     print(client3.generate(user_prompt, system_prompt))

    # else:
    #     print("Lựa chọn không hợp lệ")
    #     sys.exit(1)