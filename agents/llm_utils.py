import os
import json
import math
import re
import time
import logging
import inspect
from typing import Optional, Dict, Any, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

# Global token usage tracker for metrics
token_usage = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost": 0.0,
    "calls": [],
}


_PROVIDER_DEFAULTS = {
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "deepseek-ai/deepseek-v4-pro",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": None,
    },
}


def get_token_usage() -> Dict[str, Any]:
    return token_usage


def reset_token_usage():
    global token_usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": []}


def _provider_env_prefix(provider: str) -> str:
    """Convert an arbitrary provider label into a safe environment prefix."""
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", provider).strip("_").upper()
    return prefix or "CUSTOM"


def _resolve_llm_config(
    model: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve any OpenAI-compatible provider from environment configuration.

    Known NVIDIA, Gemini, and OpenAI providers retain convenient defaults. Any
    other provider works by setting LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, and
    either LLM_API_KEY or <PROVIDER>_API_KEY.
    """
    env = os.environ if environ is None else environ
    provider = str(env.get("LLM_PROVIDER", "")).strip().lower()

    if not provider:
        if env.get("LLM_BASE_URL") or env.get("LLM_API_KEY"):
            provider = str(env.get("LLM_PROVIDER_NAME", "custom")).strip().lower()
        else:
            # Preserve the historical NVIDIA/Gemini precedence while adding
            # first-class OpenAI auto-detection.
            provider = next(
                (
                    name
                    for name in ("nvidia", "gemini", "openai")
                    if env.get(_PROVIDER_DEFAULTS[name]["api_key_env"])
                ),
                "",
            )
    if not provider:
        raise ValueError(
            "No LLM provider is configured. Set LLM_PROVIDER plus LLM_API_KEY, "
            "LLM_BASE_URL, and LLM_MODEL, or use a supported provider-specific "
            "API key such as NVIDIA_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY."
        )

    defaults = _PROVIDER_DEFAULTS.get(provider, {})
    prefix = _provider_env_prefix(provider)
    provider_key_env = f"{prefix}_API_KEY"
    api_key = (
        env.get("LLM_API_KEY")
        or env.get(provider_key_env)
        or (env.get(defaults.get("api_key_env", "")) if defaults else None)
    )
    allow_no_key = str(env.get("LLM_ALLOW_NO_API_KEY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not api_key and allow_no_key:
        api_key = "not-required"
    if not api_key:
        raise ValueError(
            f"No API key is configured for provider {provider!r}. Set "
            f"LLM_API_KEY or {provider_key_env}; for an unauthenticated local "
            "endpoint, set LLM_ALLOW_NO_API_KEY=1."
        )

    base_url = (
        env.get("LLM_BASE_URL")
        or env.get(f"{prefix}_BASE_URL")
        or defaults.get("base_url")
    )
    if not base_url:
        raise ValueError(
            f"No OpenAI-compatible base URL is configured for provider {provider!r}. "
            "Set LLM_BASE_URL or "
            f"{prefix}_BASE_URL."
        )
    base_url = str(base_url).strip()
    if not base_url.startswith(("https://", "http://")):
        raise ValueError("LLM base URL must start with http:// or https://")

    model_name = (
        model
        or env.get("LLM_MODEL")
        or env.get(f"{prefix}_MODEL")
        or defaults.get("model")
    )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            f"No model is configured for provider {provider!r}. Set LLM_MODEL "
            f"or {prefix}_MODEL."
        )

    try:
        timeout_seconds = float(env.get("LLM_TIMEOUT_SECONDS", "120"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number")

    default_headers: Dict[str, str] = {}
    headers_json = str(env.get("LLM_DEFAULT_HEADERS_JSON", "")).strip()
    if headers_json:
        try:
            parsed_headers = json.loads(headers_json)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM_DEFAULT_HEADERS_JSON must be valid JSON") from exc
        if not isinstance(parsed_headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in parsed_headers.items()
        ):
            raise ValueError(
                "LLM_DEFAULT_HEADERS_JSON must be a JSON object of string values"
            )
        default_headers = parsed_headers

    send_temperature = str(
        env.get("LLM_SEND_TEMPERATURE", "1")
    ).strip().lower() not in {"0", "false", "no"}
    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name.strip(),
        "timeout_seconds": timeout_seconds,
        "default_headers": default_headers,
        "send_temperature": send_temperature,
    }


def call_llm(system_prompt: str, user_prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    Query any OpenAI-compatible LLM API with bounded retry handling.
    """
    global token_usage

    config = _resolve_llm_config(model=model)
    provider = config["provider"]
    model_name = config["model"]
    caller = inspect.stack()[1]
    trace_label = f"{Path(caller.filename).stem}.{caller.function}"

    # 2. Query LLM with Retry Logic for Rate Limits (429)
    retries = 5
    delay = 10.0  # Start with a 10-second delay
    
    for attempt in range(retries):
        call_started = time.monotonic()
        try:
            from openai import OpenAI
            client_kwargs = {
                "api_key": config["api_key"],
                "base_url": config["base_url"],
                "timeout": config["timeout_seconds"],
            }
            if config["default_headers"]:
                client_kwargs["default_headers"] = config["default_headers"]
            client = OpenAI(**client_kwargs)

            request = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            }
            if config["send_temperature"]:
                request["temperature"] = temperature
            response = client.chat.completions.create(**request)

            if not response.choices:
                raise ValueError(f"LLM API returned a response with an empty choices list. Response structure: {response}")
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    str(
                        item.get("text", "")
                        if isinstance(item, dict)
                        else getattr(item, "text", "")
                    )
                    for item in content
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM API returned an empty message")
            usage = getattr(response, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    in_tokens = int(
                        usage.get("prompt_tokens")
                        or usage.get("input_tokens")
                        or 0
                    )
                    out_tokens = int(
                        usage.get("completion_tokens")
                        or usage.get("output_tokens")
                        or 0
                    )
                else:
                    in_tokens = int(
                        getattr(usage, "prompt_tokens", None)
                        or getattr(usage, "input_tokens", 0)
                        or 0
                    )
                    out_tokens = int(
                        getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", 0)
                        or 0
                    )
            else:
                in_tokens = 0
                out_tokens = 0
            if in_tokens <= 0 and out_tokens <= 0:
                # Fallback token estimation
                in_tokens = int(len(system_prompt.split()) * 1.3) + int(len(user_prompt.split()) * 1.3)
                out_tokens = int(len(content.split()) * 1.3)
            token_usage["input_tokens"] += in_tokens
            token_usage["output_tokens"] += out_tokens
            token_usage["calls"].append(
                {
                    "label": trace_label,
                    "provider": provider,
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
            error_text = str(e).lower()
            is_rate_limit = False
            if hasattr(e, "status_code") and e.status_code == 429:
                is_rate_limit = True
            elif "429" in error_text or "exhausted" in error_text or "rate limit" in error_text:
                is_rate_limit = True

            if is_rate_limit and attempt < retries - 1:
                print(f"LLM Call: Hit rate limit (429). Sleeping for {delay} seconds before retry {attempt + 1}/{retries}...")
                time.sleep(delay)
                delay *= 1.5
                continue

            # Some OpenAI-compatible providers return an HTTP-400-shaped response
            # with no choices for transient server/parser faults. Retry only these
            # known transient forms; genuine invalid-request errors still fail fast.
            status_code = getattr(e, "status_code", None)
            is_transient_provider_error = (
                "empty choices" in error_text
                or "unexpected token" in error_text
                or "timed out" in error_text
                or status_code in {500, 502, 503, 504}
            )
            if is_transient_provider_error and attempt < retries - 1:
                retry_delay = min(1.0 * (2 ** attempt), 5.0)
                print(
                    "LLM Call: Transient provider response; retrying in "
                    f"{retry_delay:.1f}s ({attempt + 1}/{retries})..."
                )
                time.sleep(retry_delay)
                continue

            logger.error(f"Failed to query LLM API: {e}")
            raise e
