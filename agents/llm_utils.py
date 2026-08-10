from __future__ import annotations

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
    "calls": [],
}
_token_usage_lock = threading.Lock()
_response_cache_lock = threading.Lock()
_response_cache: "OrderedDict[str, str]" = OrderedDict()
_MAX_RESPONSE_CACHE_ENTRIES = 128
_client_cache_lock = threading.Lock()
_client_cache: dict[tuple[object, ...], object] = {}
_json_schema_capability_lock = threading.Lock()
# Structured output capability is provider-scoped: one provider without
# json_schema support must not disable it for the other providers in the process.
_json_schema_unavailable: dict[str, bool] = {}


_PROVIDER_DEFAULTS = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "deepseek-ai/deepseek-v4-flash",
    },
    "nautilus": {
        "api_key_env": "NAUT_API_KEY",
        "base_url": "https://ellm.nrp-nautilus.io/v1",
        "model": "qwen3",
    },
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "mistral": {
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
    },
}


def get_token_usage() -> Dict[str, Any]:
    return token_usage


def reset_token_usage():
    global token_usage, _json_schema_unavailable
    with _token_usage_lock:
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": [],
        }
    with _response_cache_lock:
        _response_cache.clear()
    with _client_cache_lock:
        _client_cache.clear()
    with _json_schema_capability_lock:
        _json_schema_unavailable = {}


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
    response_format: Mapping[str, object] | None = None,
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

    Known OpenAI-compatible providers are auto-detected from their API keys.
    Custom providers must declare LLM_PROVIDER/LLM_BASE_URL/LLM_MODEL so an
    unrelated service key cannot accidentally be selected as the LLM.
    """
    env = os.environ if environ is None else environ
    provider = str(env.get("LLM_PROVIDER", "")).strip().lower()

    if not provider:
        if env.get("LLM_BASE_URL") or env.get("LLM_API_KEY"):
            provider = str(env.get("LLM_PROVIDER_NAME", "custom")).strip().lower()
        else:
            for name, spec in _PROVIDER_DEFAULTS.items():
                if env.get(spec["api_key_env"]):
                    provider = name
                    break
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
        timeout_seconds = float(env.get("LLM_TIMEOUT_SECONDS", "300"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive finite number")
    try:
        max_retries = int(env.get("LLM_MAX_RETRIES", "4"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_MAX_RETRIES must be a non-negative integer") from exc
    if max_retries < 0:
        raise ValueError("LLM_MAX_RETRIES must be a non-negative integer")

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
        "max_retries": max_retries,
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


def _llm_client(config: Mapping[str, object]):
    """Reuse one HTTP client and disable the SDK's hidden retry layer."""
    from openai import OpenAI

    key = (
        config["provider"],
        config["base_url"],
        config["api_key"],
        config["timeout_seconds"],
        tuple(sorted(dict(config["default_headers"]).items())),
        OpenAI,
    )
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is not None:
            return client
        client_kwargs = {
            "api_key": config["api_key"],
            "base_url": config["base_url"],
            "timeout": config["timeout_seconds"],
            # Retrying in both the SDK and this module multiplies requests and
            # makes rate-limit recovery much worse. Keep one auditable layer.
            "max_retries": 0,
        }
        if config["default_headers"]:
            client_kwargs["default_headers"] = config["default_headers"]
        client = OpenAI(**client_kwargs)
        _client_cache[key] = client
        return client


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient_error(error: Exception) -> bool:
    status = _status_code(error)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = type(error).__name__.lower()
    text = str(error).lower()
    return any(marker in name for marker in ("connection", "timeout")) or any(
        marker in text
        for marker in (
            "connection reset",
            "connection refused",
            "empty choices",
            "empty message",
            "rate limit",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "unexpected token",
            "overloaded",
            "exhausted",
        )
    )


def _retry_delay(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = None
    if headers is not None:
        try:
            retry_after = headers.get("retry-after")
        except AttributeError:
            retry_after = None
    try:
        requested = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        requested = None
    if requested is not None and math.isfinite(requested) and requested >= 0:
        return min(requested, 60.0)
    return min(0.75 * (2 ** attempt), 12.0)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
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

    attempts = int(config["max_retries"]) + 1
    client = _llm_client(config)

    for attempt in range(attempts):
        call_started = time.monotonic()
        try:
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
                raise ValueError("LLM API returned an empty choices list")
            msg = response.choices[0].message
            content = msg.content
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
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "thinking", None)
                if reasoning and isinstance(reasoning, str) and reasoning.strip():
                    content = reasoning

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
            
        except Exception as error:
            if _is_transient_error(error) and attempt < attempts - 1:
                retry_delay = _retry_delay(error, attempt)
                print(
                    "LLM Call: Transient provider response; retrying in "
                    f"{retry_delay:.1f}s ({attempt + 1}/{attempts - 1})..."
                )
                time.sleep(retry_delay)
                continue

            if response_format is not None and _response_format_is_unavailable(error):
                logger.info("Provider does not support this response format: %s", error)
            else:
                logger.error("Failed to query LLM API: %s", error)
            raise


def _extract_json_payload(content: str) -> object:
    text = str(content or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text)
        return value
    except json.JSONDecodeError:
        pass
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response did not contain a valid JSON value")


def _response_format_is_unavailable(error: Exception) -> bool:
    text = str(error).casefold()
    return "response_format" in text and any(
        marker in text
        for marker in (
            "unavailable",
            "unsupported",
            "not available",
            "not supported",
            "unknown field",
        )
    )


def _should_try_json_schema(provider: str | None = None) -> bool:
    configured = os.getenv("LLM_USE_JSON_SCHEMA", "auto").strip().casefold()
    if configured in {"0", "false", "no", "off"}:
        return False
    if provider is None:
        try:
            provider = _resolve_llm_config()["provider"]
        except Exception:
            provider = "unknown"
    with _json_schema_capability_lock:
        return not _json_schema_unavailable.get(provider, False)


def _remember_json_schema_unavailable(provider: str) -> None:
    global _json_schema_unavailable
    with _json_schema_capability_lock:
        _json_schema_unavailable[provider] = True


def _evict_cached_response(key: str) -> None:
    """Drop one cached response so a rejected JSON value cannot poison retries."""
    with _response_cache_lock:
        _response_cache.pop(key, None)


def _validate_json_schema(value: object, schema: Mapping[str, object], path: str = "$") -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        required = schema.get("required", [])
        if isinstance(required, (list, tuple)):
            for key in required:
                if key not in value:
                    raise ValueError(f"{path} is missing required field {key!r}")
        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            if schema.get("additionalProperties") is False:
                unexpected = sorted(set(value) - set(properties))
                if unexpected:
                    raise ValueError(
                        f"{path} contains unexpected field(s): {', '.join(unexpected)}"
                    )
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    _validate_json_schema(value[key], child_schema, f"{path}.{key}")
    elif expected == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        minimum = int(str(schema.get("minItems", 0) or 0))
        maximum = schema.get("maxItems")
        if len(value) < minimum or (
            maximum is not None and len(value) > int(str(maximum))
        ):
            raise ValueError(f"{path} has the wrong item count")
        child_schema = schema.get("items")
        if isinstance(child_schema, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema(item, child_schema, f"{path}[{index}]")
    elif expected == "string" and not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    elif expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
        if "minimum" in schema and value < int(str(schema["minimum"])):
            raise ValueError(f"{path} is below its minimum")
        if "maximum" in schema and value > int(str(schema["maximum"])):
            raise ValueError(f"{path} is above its maximum")
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite")
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple, set)) and value not in enum:
        raise ValueError(f"{path} contains a value outside its enum")


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    schema: Mapping[str, object],
    schema_name: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> object:
    """Return schema-checked JSON, repairing providers without response formats."""
    normalized_name = re.sub(r"[^A-Za-z0-9_-]+", "_", schema_name)[:64] or "response"
    structured_format = {
        "type": "json_schema",
        "json_schema": {
            "name": normalized_name,
            "strict": True,
            "schema": dict(schema),
        },
    }
    try:
        provider = _resolve_llm_config(model)["provider"]
    except Exception:
        provider = "unknown"
    response: str | None = None
    used_structured = False
    if _should_try_json_schema(provider):
        try:
            response = call_llm(
                system_prompt,
                user_prompt,
                model=model,
                temperature=temperature,
                response_format=structured_format,
            )
            used_structured = True
        except Exception as structured_error:
            if _response_format_is_unavailable(structured_error):
                _remember_json_schema_unavailable(provider)
            logger.warning(
                "Provider rejected structured output for %s; using JSON-only fallback: %s",
                normalized_name,
                structured_error,
            )

    schema_text = json.dumps(schema, indent=2, ensure_ascii=False, default=str)
    if response is None:
        fallback_prompt = f"""
{user_prompt}

OUTPUT CONTRACT:
Return exactly one JSON value and no prose or Markdown. It must validate against
this JSON Schema:
{schema_text}
""".strip()
        response = call_llm(
            system_prompt + " Return only the requested JSON value.",
            fallback_prompt,
            model=model,
            temperature=temperature,
        )

    def evict_poisoned() -> None:
        # A schema-invalid response was cached under the exact request key; remove
        # it so the next identical call does not re-enter the repair loop.
        try:
            resolved = _resolve_llm_config(model)
            key = _request_cache_key(
                provider=str(resolved["provider"]),
                model=str(resolved["model"]),
                system_prompt=(
                    system_prompt
                    if used_structured
                    else system_prompt + " Return only the requested JSON value."
                ),
                user_prompt=user_prompt if used_structured else fallback_prompt,
                temperature=float(temperature),
                response_format=structured_format if used_structured else None,
            )
            _evict_cached_response(key)
        except Exception:
            pass

    try:
        payload = _extract_json_payload(response)
        _validate_json_schema(payload, schema)
        return payload
    except (TypeError, ValueError, json.JSONDecodeError) as validation_error:
        evict_poisoned()
        repair_prompt = f"""
Repair the JSON response below. Return exactly one corrected JSON value with no
prose or Markdown.

Validation error:
{validation_error}

Required JSON Schema:
{schema_text}

Invalid response:
{response[:24000]}
""".strip()
        repaired = call_llm(
            "You repair structured JSON without changing its intended factual content.",
            repair_prompt,
            model=model,
            temperature=0.0,
        )
        payload = _extract_json_payload(repaired)
        _validate_json_schema(payload, schema)
        return payload
