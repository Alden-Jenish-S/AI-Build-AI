"""
Web Search Module — Robust, zero-dependency web search for the agent system.

Provides search_web(query) which tries multiple strategies in order:
  1. Serper API (if SERPER_API_KEY is set)
  2. DuckDuckGo Lite HTML scraping (with retries + user-agent rotation)
  3. DuckDuckGo JSON API fallback (instant answers endpoint)

All methods use only the Python stdlib (urllib, json, re) — no pip installs needed.
"""

import json
import os
import re
import time
import random
import urllib.request
import urllib.parse
import http.client
from typing import List, Optional

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


def _clean_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text).strip()


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

            # Parse the lite results page
            # DDG Lite uses a table-based layout with <a class="result-link"> for links
            # and <td class="result-snippet"> for snippets
            results = []

            # Try multiple parsing strategies
            # Strategy A: Look for result-link anchors
            link_matches = re.findall(
                r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )
            snippet_matches = re.findall(
                r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                html, re.DOTALL
            )

            if link_matches:
                for i, (link, title_html) in enumerate(link_matches[:5]):
                    title = _clean_html(title_html)
                    snippet = _clean_html(snippet_matches[i]) if i < len(snippet_matches) else ""
                    # Decode DDG redirect URLs
                    if "uddg=" in link:
                        try:
                            qs = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                            if 'uddg' in qs:
                                link = qs['uddg'][0]
                        except Exception:
                            pass
                    results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n")

            # Strategy B: Fallback to generic link/snippet extraction
            if not results:
                # Look for any links that look like search results
                all_links = re.findall(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
                for link, title_html in all_links[:5]:
                    # Skip DDG internal links
                    if 'duckduckgo.com' in link:
                        continue
                    title = _clean_html(title_html)
                    if title and len(title) > 5:
                        results.append(f"Title: {title}\nURL: {link}\nSnippet: \n")

            if results:
                return "\n".join(results)

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

            divs = html.split('<div class="result')
            results = []
            for div in divs[1:6]:
                link_match = re.search(r'href="([^"]+)"', div)
                link = link_match.group(1) if link_match else ""
                if link.startswith("//"):
                    link = "https:" + link
                if "uddg=" in link:
                    try:
                        qs = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        if 'uddg' in qs:
                            link = qs['uddg'][0]
                    except Exception:
                        pass

                title_match = re.search(r'<a class="result__url"[^>]*>(.*?)</a>', div, re.DOTALL)
                title = _clean_html(title_match.group(1)) if title_match else "No Title"

                snippet_match = re.search(r'<a class="result__snippet"[^>]*>(.*?)</a>', div, re.DOTALL)
                snippet = _clean_html(snippet_match.group(1)) if snippet_match else ""

                results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n")

            if results:
                return "\n".join(results)

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

def search_web(query_str: str) -> str:
    """
    Search the web using the best available strategy.

    Tries in order:
      1. Serper API (if SERPER_API_KEY env var is set)
      2. DuckDuckGo Lite (POST-based, with retries + UA rotation)
      3. DuckDuckGo HTML (GET-based, with retries + UA rotation)
      4. DuckDuckGo Instant Answer JSON API (limited but never blocked)
      5. Empty fallback message

    Returns formatted search results as a string.
    """
    # 1. Serper (Google)
    result = _serper_search(query_str)
    if result:
        print("[WebSearch] Results from Serper API (Google)")
        return result

    # 2. DDG Lite
    result = _ddg_lite_search(query_str)
    if result:
        print("[WebSearch] Results from DuckDuckGo Lite")
        return result

    # 3. DDG HTML
    result = _ddg_html_search(query_str)
    if result:
        print("[WebSearch] Results from DuckDuckGo HTML")
        return result

    # 4. DDG Instant Answer
    result = _ddg_instant_answer(query_str)
    if result:
        print("[WebSearch] Results from DuckDuckGo Instant Answer API")
        return result

    # 5. All strategies failed
    print("[WebSearch] WARNING: All search strategies failed!")
    return ""
