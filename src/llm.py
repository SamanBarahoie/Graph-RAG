# src/llm.py
import requests
from typing import Optional


API_BASE_URL = "https://openrouter.ai/api/v1"


class LLM:
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url or API_BASE_URL
        self.model = model


    def generate_response(self, prompt: str, system: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json"
        }
        data = {
        "model": self.model,
        "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
        }


        resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()