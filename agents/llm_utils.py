import os
import time
import logging
import inspect
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Global token usage tracker for metrics
token_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost": 0.0,
    "calls": [],
}

def get_token_usage() -> Dict[str, Any]:
    return token_usage

def reset_token_usage():
    global token_usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": []}

def call_llm(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    Queries the LLM API with exponential backoff on 429 rate limit errors.
    Dynamically supports both NVIDIA API and Gemini API.
    """
    global token_usage
    
    # 1. Resolve API Key & Base URL
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in {"", "nvidia", "gemini"}:
        raise ValueError("LLM_PROVIDER must be either 'nvidia' or 'gemini'")

    if provider == "nvidia" or (not provider and os.getenv("NVIDIA_API_KEY")):
        api_key = os.getenv("NVIDIA_API_KEY")
        base_url = "https://integrate.api.nvidia.com/v1"
        provider_default_model = "openai/gpt-oss-120b"
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        provider_default_model = "gemini-2.5-flash"

    if not api_key:
        raise ValueError(
            "The selected LLM provider has no API key. Set NVIDIA_API_KEY or GEMINI_API_KEY."
        )
        
    model_name = model or os.getenv("LLM_MODEL") or provider_default_model
    caller = inspect.stack()[1]
    trace_label = f"{Path(caller.filename).stem}.{caller.function}"
    
    # 2. Query LLM with Retry Logic for Rate Limits (429)
    retries = 5
    delay = 10.0  # Start with a 10-second delay
    
    for attempt in range(retries):
        call_started = time.monotonic()
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
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM API returned an empty message")
            usage = response.usage
            if usage:
                in_tokens = int(usage.prompt_tokens)
                out_tokens = int(usage.completion_tokens)
            else:
                # Fallback token estimation
                in_tokens = int(len(system_prompt.split()) * 1.3) + int(len(user_prompt.split()) * 1.3)
                out_tokens = int(len(content.split()) * 1.3)
            token_usage["input_tokens"] += in_tokens
            token_usage["output_tokens"] += out_tokens
            token_usage["calls"].append(
                {
                    "label": trace_label,
                    "provider": provider or ("nvidia" if os.getenv("NVIDIA_API_KEY") else "gemini"),
                    "model": model_name,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "system_prompt_chars": len(system_prompt),
                    "user_prompt_chars": len(user_prompt),
                    "elapsed_seconds": time.monotonic() - call_started,
                }
            )
                
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
