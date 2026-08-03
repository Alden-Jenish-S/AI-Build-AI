import os
import json
import math
import re
import time
import logging
import inspect
import hashlib
import threading
from collections import OrderedDict
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
_token_usage_lock = threading.Lock()
_response_cache_lock = threading.Lock()
_response_cache: "OrderedDict[str, str]" = OrderedDict()
_MAX_RESPONSE_CACHE_ENTRIES = 128


_PROVIDER_DEFAULTS = {
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "openai/gpt-oss-120b",
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
    with _token_usage_lock:
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "calls": [],
        }
    with _response_cache_lock:
        _response_cache.clear()


def _cache_flag(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _request_cache_key(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    response_format: Mapping[str, object] | None,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": float(temperature),
        "response_format": response_format,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cached_response(key: str) -> str | None:
    with _response_cache_lock:
        response = _response_cache.get(key)
        if response is not None:
            _response_cache.move_to_end(key)
        return response


def _remember_response(key: str, response: str) -> None:
    with _response_cache_lock:
        _response_cache[key] = response
        _response_cache.move_to_end(key)
        while len(_response_cache) > _MAX_RESPONSE_CACHE_ENTRIES:
            _response_cache.popitem(last=False)


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
        "cache_deterministic_responses": _cache_flag(
            env.get("LLM_CACHE_DETERMINISTIC_RESPONSES"), default=True
        ),
        "send_prompt_cache_key": _cache_flag(
            env.get("LLM_SEND_PROMPT_CACHE_KEY"),
            default=provider == "openai",
        ),
    }


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    *,
    response_format: Mapping[str, object] | None = None,
) -> str:
    """
    Query any OpenAI-compatible LLM API with bounded retry handling.
    """
    global token_usage

    config = _resolve_llm_config(model=model)
    provider = config["provider"]
    model_name = config["model"]
    caller = inspect.stack()[1]
    trace_label = f"{Path(caller.filename).stem}.{caller.function}"
    cache_key = _request_cache_key(
        provider=provider,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        response_format=response_format,
    )
    use_response_cache = bool(
        config["cache_deterministic_responses"]
        and float(temperature) == 0.0
    )
    if use_response_cache:
        cached = _cached_response(cache_key)
        if cached is not None:
            with _token_usage_lock:
                token_usage["calls"].append(
                    {
                        "label": trace_label,
                        "provider": provider,
                        "model": model_name,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "system_prompt_chars": len(system_prompt),
                        "user_prompt_chars": len(user_prompt),
                        "elapsed_seconds": 0.0,
                        "cache_hit": True,
                    }
                )
            return cached

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
            if response_format is not None:
                request["response_format"] = dict(response_format)
            if config["send_prompt_cache_key"]:
                request["extra_body"] = {
                    "prompt_cache_key": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest()
                }
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
            with _token_usage_lock:
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
                        "cache_hit": False,
                    }
                )
            if use_response_cache:
                _remember_response(cache_key, content)
                
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


def _extract_json_payload(content: str) -> object:
    text = str(content).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    decoder = json.JSONDecoder()
    # Structured providers normally return a complete JSON value. Decode from
    # the beginning first so an outer array is not mistaken for its first
    # object element. The fallback supports gateways that prefix brief prose.
    try:
        payload, _ = decoder.raw_decode(text)
        return payload
    except json.JSONDecodeError:
        pass
    starts = sorted(
        index for index, character in enumerate(text) if character in "[{"
    )
    for start in starts:
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response did not contain a valid JSON value")


def _validate_json_schema(value: object, schema: Mapping[str, object]) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError("structured LLM response must be a JSON object")
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"structured LLM response is missing {key!r}")
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    _validate_json_schema(value[key], child_schema)
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError("structured LLM response must be a JSON array")
        minimum = int(schema.get("minItems", 0) or 0)
        maximum = schema.get("maxItems")
        if len(value) < minimum or (
            maximum is not None and len(value) > int(maximum)
        ):
            raise ValueError("structured LLM response has the wrong item count")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for item in value:
                _validate_json_schema(item, item_schema)
    elif expected_type == "string" and not isinstance(value, str):
        raise ValueError("structured LLM response field must be a string")
    elif expected_type == "boolean" and not isinstance(value, bool):
        raise ValueError("structured LLM response field must be boolean")
    elif expected_type == "number" and not isinstance(value, (int, float)):
        raise ValueError("structured LLM response field must be numeric")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("structured LLM response contains an invalid enum value")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    schema: Mapping[str, object],
    schema_name: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> object:
    """Return one schema-constrained JSON response without repair prompting."""
    response = call_llm(
        system_prompt,
        user_prompt,
        model=model,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": re.sub(r"[^A-Za-z0-9_-]+", "_", schema_name)[:64],
                "strict": True,
                "schema": dict(schema),
            },
        },
    )
    payload = _extract_json_payload(response)
    _validate_json_schema(payload, schema)
    return payload
