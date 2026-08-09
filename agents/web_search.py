"""
Web Search Module — Robust, zero-dependency web search for the agent system.

Provides search_web(query) which tries multiple strategies in order:
  1. Serper API (if SERPER_API_KEY is set)
  2. DuckDuckGo Lite HTML scraping (with retries + user-agent rotation)
  3. DuckDuckGo HTML scraping
  4. DuckDuckGo JSON API fallback (instant answers endpoint)

All methods use only the Python stdlib. Search-page HTML is parsed structurally
with ``html.parser`` rather than regular expressions tied to CSS class names.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
import os
import re
import time
import random
import urllib.request
import urllib.parse
import http.client
import threading
from typing import Optional


# Preserve precise, source-targeted literature queries. The previous 99-character
# cap silently removed the discriminating tail of council-generated queries.
_MAX_QUERY_CHARS = 300
_MAX_CACHE_ENTRIES = 256
_SEARCH_CACHE: dict[str, str] = {}
_SEARCH_CACHE_LOCK = threading.RLock()
_SEARCH_INFLIGHT: dict[str, threading.Event] = {}

# Pool of realistic browser user-agents to rotate through
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


def _random_ua() -> str:
    """Return a random user-agent string."""
    return random.choice(_USER_AGENTS)


def _bounded_query(query: object) -> str:
    """Normalize a query while retaining its technical qualifiers."""
    normalized = " ".join(str(query or "").split())
    if len(normalized) <= _MAX_QUERY_CHARS:
        return normalized
    prefix = normalized[:_MAX_QUERY_CHARS]
    if " " in prefix:
        shortened = prefix.rsplit(" ", 1)[0].strip()
        if shortened:
            return shortened
    return prefix


def clear_search_cache() -> None:
    """Clear process-local search results, primarily for run/test boundaries."""
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE.clear()


def _result_url(href: str) -> str | None:
    """Decode a DDG redirect or accept a direct external HTTP(S) URL."""
    raw = str(href or "").strip()
    if not raw or raw.startswith(("#", "javascript:", "mailto:")):
        return None
    parse_target = "https:" + raw if raw.startswith("//") else raw
    if raw.startswith("/"):
        parse_target = "https://duckduckgo.com" + raw
    try:
        parsed = urllib.parse.urlparse(parse_target)
        query = urllib.parse.parse_qs(parsed.query)
        redirected = query.get("uddg", [None])[0]
        if redirected:
            parse_target = urllib.parse.unquote(str(redirected))
            parsed = urllib.parse.urlparse(parse_target)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        host = (parsed.hostname or "").lower()
        if host == "duckduckgo.com" or host.endswith(".duckduckgo.com"):
            return None
        return parse_target
    except (TypeError, ValueError):
        return None


class _SearchPageParser(HTMLParser):
    """Extract external result anchors and their nearby visible text."""

    def __init__(self, limit: int = 5) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[dict[str, object]] = []
        self._hidden_depth = 0
        self._inside_anchor = False
        self._anchor_url: str | None = None
        self._anchor_text: list[str] = []
        self._active_result: int | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "form"}:
            self._hidden_depth += 1
            return
        if lowered != "a" or self._hidden_depth:
            return
        self._inside_anchor = True
        attributes = {str(name).lower(): value for name, value in attrs}
        self._anchor_url = _result_url(str(attributes.get("href") or ""))
        self._anchor_text = []

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript", "form"}:
            self._hidden_depth = max(0, self._hidden_depth - 1)
            return
        if lowered != "a" or not self._inside_anchor:
            return
        if self._anchor_url:
            title = " ".join(self._anchor_text).strip()
            if title:
                existing_index = next(
                    (
                        index
                        for index, item in enumerate(self.results)
                        if str(item["url"]) == self._anchor_url
                    ),
                    None,
                )
                if existing_index is None and len(self.results) < self.limit:
                    self.results.append(
                        {
                            "title": title,
                            "url": self._anchor_url,
                            "snippet_parts": [],
                        }
                    )
                    self._active_result = len(self.results) - 1
                elif existing_index is not None:
                    parts = self.results[existing_index]["snippet_parts"]
                    if (
                        isinstance(parts, list)
                        and title != self.results[existing_index]["title"]
                    ):
                        parts.append(title)
                    self._active_result = existing_index
        self._inside_anchor = False
        self._anchor_url = None
        self._anchor_text = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned or self._hidden_depth:
            return
        if self._inside_anchor:
            if self._anchor_url:
                self._anchor_text.append(cleaned)
            return
        if self._active_result is None:
            return
        parts = self.results[self._active_result]["snippet_parts"]
        if not isinstance(parts, list):
            return
        joined_length = sum(len(str(item)) for item in parts)
        if joined_length < 500 and cleaned.lower() not in {
            "more results",
            "feedback",
            "next",
        }:
            parts.append(cleaned)


def _parse_search_html(html: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Parse result pages without relying on provider CSS class names."""
    parser = _SearchPageParser(limit=limit)
    parser.feed(html)
    parser.close()
    parsed = []
    for result in parser.results:
        title = " ".join(str(result["title"]).split())
        snippet = " ".join(
            str(item)
            for item in result.get("snippet_parts", [])
            if str(item).strip()
        )
        # Stop generic nearby-text capture before the next result title when it
        # was emitted outside its anchor by unusual markup.
        parsed.append(
            {
                "title": title[:300],
                "url": str(result["url"]),
                "snippet": snippet[:600],
            }
        )
    return parsed


def _format_results(results: list[dict[str, str]]) -> str | None:
    rendered = [
        f"Title: {item['title']}\nURL: {item['url']}\n"
        f"Snippet: {item.get('snippet', '')}\n"
        for item in results
        if item.get("title") and item.get("url")
    ]
    return "\n".join(rendered) if rendered else None


# ---------------------------------------------------------------------------
# Strategy 1: Serper API (Google results, needs SERPER_API_KEY env var)
# ---------------------------------------------------------------------------

def _serper_search(query_str: str) -> Optional[str]:
    """Query Google via Serper API. Returns formatted results or None on failure."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return None

    try:
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=15)
        payload = json.dumps({"q": query_str})
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse().read().decode("utf-8")
        data = json.loads(res)

        out = []
        for item in data.get("organic", [])[:5]:
            out.append(
                f"Title: {item.get('title', 'No Title')}\n"
                f"URL: {item.get('link', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
            )
        result = "\n".join(out)
        if result.strip():
            return result
        return None
    except Exception as e:
        print(f"[WebSearch] Serper API failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Strategy 2: DuckDuckGo Lite HTML Scraping (with retries + UA rotation)
# ---------------------------------------------------------------------------

def _ddg_lite_search(query_str: str, max_retries: int = 3) -> Optional[str]:
    """
    Scrape DuckDuckGo Lite (lite.duckduckgo.com) which is a minimal HTML page
    designed for text browsers — much less likely to be blocked than html.duckduckgo.com.
    Retries with different user-agents and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            # Add jitter delay between retries (0s on first attempt)
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0.5, 2.0)
                print(f"[WebSearch] DDG Lite retry {attempt + 1}/{max_retries} after {delay:.1f}s...")
                time.sleep(delay)

            # Use POST to lite.duckduckgo.com (mimics form submission, harder to block)
            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({"q": query_str}).encode("utf-8")
            headers = {
                "User-Agent": _random_ua(),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://lite.duckduckgo.com/",
            }

            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="replace")

            results = _format_results(_parse_search_html(html))
            if results:
                return results

        except Exception as e:
            print(f"[WebSearch] DDG Lite attempt {attempt + 1} failed: {e}")

    return None


# ---------------------------------------------------------------------------
# Strategy 3: DuckDuckGo HTML (original, as secondary fallback)
# ---------------------------------------------------------------------------

def _ddg_html_search(query_str: str, max_retries: int = 2) -> Optional[str]:
    """Scrape DuckDuckGo HTML endpoint with retries and UA rotation."""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2.0 + random.uniform(0.5, 1.5))

            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_str)}"
            headers = {
                "User-Agent": _random_ua(),
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="replace")

            results = _format_results(_parse_search_html(html))
            if results:
                return results

        except Exception as e:
            print(f"[WebSearch] DDG HTML attempt {attempt + 1} failed: {e}")

    return None


# ---------------------------------------------------------------------------
# Strategy 4: DuckDuckGo Instant Answer JSON API (limited but never blocked)
# ---------------------------------------------------------------------------

def _ddg_instant_answer(query_str: str) -> Optional[str]:
    """
    Query the DuckDuckGo Instant Answer API (api.duckduckgo.com).
    This returns structured data, not full web results. It's limited to
    topics DDG has instant answers for, but it's completely free and never blocked.
    """
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query_str)}&format=json&no_html=1&skip_disambig=1"
        headers = {"User-Agent": _random_ua()}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []

        # Abstract (main topic summary)
        if data.get("AbstractText"):
            results.append(
                f"Title: {data.get('Heading', 'Topic')}\n"
                f"URL: {data.get('AbstractURL', '')}\n"
                f"Snippet: {data['AbstractText'][:300]}\n"
            )

        # Related topics
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    f"Title: {topic.get('Text', '')[:80]}\n"
                    f"URL: {topic.get('FirstURL', '')}\n"
                    f"Snippet: {topic.get('Text', '')[:200]}\n"
                )

        if results:
            return "\n".join(results)
        return None
    except Exception as e:
        print(f"[WebSearch] DDG Instant Answer failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def _search_uncached(query: str, *, max_retries: int | None = None) -> str:
    """Run provider fallbacks for one already-normalized query."""
    lite_retries = 3 if max_retries is None else max_retries
    html_retries = 2 if max_retries is None else max_retries
    # 1. Serper (Google)
    result = _serper_search(query)
    if result:
        print("[WebSearch] Results from Serper API (Google)")
        return result

    # 2. DDG Lite
    result = _ddg_lite_search(query, max_retries=lite_retries)
    if result:
        print("[WebSearch] Results from DuckDuckGo Lite")
        return result

    # 3. DDG HTML
    result = _ddg_html_search(query, max_retries=html_retries)
    if result:
        print("[WebSearch] Results from DuckDuckGo HTML")
        return result

    # 4. DDG Instant Answer
    result = _ddg_instant_answer(query)
    if result:
        print("[WebSearch] Results from DuckDuckGo Instant Answer API")
        return result

    # 5. All strategies failed
    print("[WebSearch] WARNING: All search strategies failed!")
    return ""


def search_web(query_str: str, *, max_retries: int | None = None) -> str:
    """Search with bounded queries, per-process caching, and request coalescing."""
    query = _bounded_query(query_str)
    if not query:
        return ""
    if max_retries is not None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("max_retries must be a positive integer or None")
        max_retries = max(1, min(5, max_retries))
    cache_key = query.casefold()
    with _SEARCH_CACHE_LOCK:
        if cache_key in _SEARCH_CACHE:
            print("[WebSearch] Results from in-memory cache")
            return _SEARCH_CACHE[cache_key]
        pending = _SEARCH_INFLIGHT.get(cache_key)
        if pending is None:
            pending = threading.Event()
            _SEARCH_INFLIGHT[cache_key] = pending
            owns_request = True
        else:
            owns_request = False

    if not owns_request:
        pending.wait(timeout=75)
        with _SEARCH_CACHE_LOCK:
            return _SEARCH_CACHE.get(cache_key, "")

    try:
        return _cache_result(
            cache_key, _search_uncached(query, max_retries=max_retries)
        )
    finally:
        with _SEARCH_CACHE_LOCK:
            completed = _SEARCH_INFLIGHT.pop(cache_key, None)
            if completed is not None:
                completed.set()


def _cache_result(cache_key: str, result: str) -> str:
    """Store one result with a small process-local size bound."""
    # A temporary provider/rate-limit failure must not poison this query for
    # the rest of a long search run.
    if not result:
        return ""
    with _SEARCH_CACHE_LOCK:
        if len(_SEARCH_CACHE) >= _MAX_CACHE_ENTRIES:
            oldest_key = next(iter(_SEARCH_CACHE))
            _SEARCH_CACHE.pop(oldest_key, None)
        _SEARCH_CACHE[cache_key] = result
    return result
