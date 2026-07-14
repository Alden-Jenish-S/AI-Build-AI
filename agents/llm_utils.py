import os
import sys
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Centralized Model Selection - Change this value to switch model across all agents
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Global token usage tracker for metrics
token_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost": 0.0
}

def get_token_usage() -> Dict[str, Any]:
    return token_usage

def reset_token_usage():
    global token_usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

def call_llm(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    Queries the LLM API with exponential backoff on 429 rate limit errors.
    Dynamically supports both NVIDIA API and Gemini API.
    """
    global token_usage
    
    # 1. Resolve API Key & Base URL
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is not configured! Please set either 'NVIDIA_API_KEY' or 'GEMINI_API_KEY' environment variable."
        )
        
    if os.getenv("NVIDIA_API_KEY"):
        base_url = "https://integrate.api.nvidia.com/v1"
        provider_default_model = "nvidia/nemotron-3-ultra-550b-a55b"
    else:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        provider_default_model = "gemini-2.5-flash"
        
    model_name = model or os.getenv("LLM_MODEL") or provider_default_model
    
    # 2. Query LLM with Retry Logic for Rate Limits (429)
    retries = 5
    delay = 10.0  # Start with a 10-second delay
    
    for attempt in range(retries):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            if not response.choices:
                raise ValueError(f"LLM API returned a response with an empty choices list. Response structure: {response}")
            content = response.choices[0].message.content
            usage = response.usage
            if usage:
                token_usage["input_tokens"] += usage.prompt_tokens
                token_usage["output_tokens"] += usage.completion_tokens
            else:
                # Fallback token estimation
                in_tokens = int(len(system_prompt.split()) * 1.3) + int(len(user_prompt.split()) * 1.3)
                out_tokens = int(len(content.split()) * 1.3)
                token_usage["input_tokens"] += in_tokens
                token_usage["output_tokens"] += out_tokens
                
            return content
            
        except Exception as e:
            # Check if this is a rate limit error (status code 429 or matching string)
            is_rate_limit = False
            if hasattr(e, "status_code") and e.status_code == 429:
                is_rate_limit = True
            elif "429" in str(e) or "exhausted" in str(e).lower() or "rate limit" in str(e).lower():
                is_rate_limit = True
                
            if is_rate_limit and attempt < retries - 1:
                print(f"LLM Call: Hit rate limit (429). Sleeping for {delay} seconds before retry {attempt + 1}/{retries}...")
                time.sleep(delay)
                delay *= 1.5
            else:
                logger.error(f"Failed to query LLM API: {e}")
                raise e
