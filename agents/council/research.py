"""De-identified, provenance-preserving academic research retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from ..web_search import search_web


_PROHIBITED_QUERY_PHRASES = {
    "kaggle",
    "winning solution",
    "winner solution",
    "competition solution",
    "leaderboard solution",
    "public notebook",
    "private leaderboard",
}
_PROHIBITED_HOSTS = {
    "kaggle.com",
    "www.kaggle.com",
    "medium.com",
    "towardsdatascience.com",
}
_PRIMARY_HOST_SUFFIXES = (
    "arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
    "jmlr.org",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "openaccess.thecvf.com",
    "aclanthology.org",
    "ojs.aaai.org",
    "biorxiv.org",
    "medrxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ieee.org",
    "acm.org",
    "springer.com",
    "nature.com",
    "science.org",
    "research.google",
    "microsoft.com",
    "meta.com",
)
_OPENALEX_RATE_LOCK = threading.Lock()
_OPENALEX_LAST_REQUEST = 0.0


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() in {"script", "style", "nav", "header", "footer", "form"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "nav", "header", "footer", "form"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text and not self.hidden:
            self.parts.append(text)


def _task_tokens(task_name: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", str(task_name))
        if len(token) >= 4 and token.casefold() not in {"classification", "series", "challenge"}
    }


def validate_research_query(
    query: object,
    *,
    task_name: str,
    forbidden_terms: Iterable[str] = (),
) -> tuple[bool, str]:
    normalized = " ".join(str(query or "").split())
    lowered = normalized.casefold()
    if not normalized:
        return False, "query is empty"
    if len(normalized) > 300:
        return False, "query exceeds 300 characters"
    phrase = next((item for item in _PROHIBITED_QUERY_PHRASES if item in lowered), None)
    if phrase:
        return False, f"query contains prohibited phrase {phrase!r}"
    protected = _task_tokens(task_name) | {
        str(term).strip().casefold()
        for term in forbidden_terms
        if len(str(term).strip()) >= 4
    }
    leaked = sorted(token for token in protected if token in lowered)
    if leaked:
        return False, "query contains task-identity token(s): " + ", ".join(leaked)
    content_terms = [
        term
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", normalized)
        if term.casefold() not in {"site", "and", "or", "the", "for", "with", "machine", "learning"}
    ]
    if len(content_terms) < 4:
        return False, "query is too broad; fewer than four technical terms"
    if "site:" not in lowered and '"' not in normalized:
        return False, "query must target a source domain or contain a precise quoted concept"
    return True, "accepted"


def _parse_formatted_results(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in str(text or "").splitlines():
        if line.startswith("Title: "):
            if current.get("title") and current.get("url"):
                records.append(current)
            current = {"title": line[7:].strip()}
        elif line.startswith("URL: "):
            current["url"] = line[5:].strip()
        elif line.startswith("Snippet: "):
            current["snippet"] = line[9:].strip()
        elif current.get("snippet") and line.strip():
            current["snippet"] = (current["snippet"] + " " + line.strip())[:2000]
    if current.get("title") and current.get("url"):
        records.append(current)
    return records


def _host_allowed(url: str) -> tuple[bool, str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False, "invalid URL"
    host = (parsed.hostname or "").casefold()
    if not host or parsed.scheme not in {"http", "https"}:
        return False, "not an HTTP(S) URL"
    if host in _PROHIBITED_HOSTS or any(host.endswith("." + item) for item in _PROHIBITED_HOSTS):
        return False, "competition solution or non-primary source host"
    if not any(host == suffix or host.endswith("." + suffix) for suffix in _PRIMARY_HOST_SUFFIXES):
        return False, "source is outside the primary/authoritative domain allowlist"
    return True, "accepted"


def _fetch_visible_text(url: str, limit: int = 80_000) -> str:
    headers = {
        "User-Agent": "AIBuildAI-ResearchCouncil/1.0 (+academic literature retrieval)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        content_type = str(response.headers.get("content-type", "")).casefold()
        raw = response.read(min(limit * 3, 1_000_000))
    if "pdf" in content_type or url.casefold().endswith(".pdf"):
        return ""
    text = raw.decode("utf-8", errors="replace")
    if "html" not in content_type and "<html" not in text[:1000].casefold():
        return " ".join(text.split())[:limit]
    parser = _VisibleTextParser()
    parser.feed(text)
    parser.close()
    return " ".join(parser.parts)[:limit]


def _openalex_terms(query: str) -> tuple[str, str | None]:
    """Translate a source-targeted web query into OpenAlex text and date filters."""
    current_year = datetime.now(timezone.utc).year
    years = [
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", query)
        if 1900 <= int(value) <= current_year + 1
    ]
    text = re.sub(r"(?i)\bsite:\S+", " ", query)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = text.replace('"', " ")
    text = re.sub(r"(?i)\b(?:AND|OR)\b", " ", text)
    text = " ".join(text.split())[:1000]
    if not text:
        return "", None
    if years:
        return (
            text,
            f"from_publication_date:{min(years)}-01-01,"
            f"to_publication_date:{max(years)}-12-31",
        )
    if "foundational" in query.casefold():
        return text, None
    return (
        text,
        f"from_publication_date:{current_year - 3}-01-01,"
        f"to_publication_date:{current_year}-12-31",
    )


def _abstract_from_inverted_index(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for token, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and not isinstance(position, bool):
                positioned.append((position, str(token)))
    positioned.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positioned)


def _openalex_search(query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Search OpenAlex without ever persisting or logging its API key."""
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if not api_key:
        return []
    terms, date_filter = _openalex_terms(query)
    if not terms:
        return []
    parameters = {
        "api_key": api_key,
        "search": terms,
        "per_page": str(max(1, min(10, int(limit)))),
        "select": (
            "id,doi,title,display_name,publication_year,publication_date,"
            "primary_location,best_oa_location,open_access,abstract_inverted_index,"
            "cited_by_count,type,authorships"
        ),
    }
    if date_filter:
        parameters["filter"] = date_filter
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "AIBuildAI-ResearchCouncil/1.0",
        },
    )
    try:
        timeout = max(
            3.0, min(60.0, float(os.getenv("AIBUILDAI_OPENALEX_TIMEOUT_SECONDS", "20")))
        )
    except ValueError:
        timeout = 20.0
    try:
        interval = max(
            0.0,
            min(
                5.0,
                float(os.getenv("AIBUILDAI_OPENALEX_MIN_INTERVAL_SECONDS", "1.0")),
            ),
        )
    except ValueError:
        interval = 1.0
    global _OPENALEX_LAST_REQUEST
    with _OPENALEX_RATE_LOCK:
        remaining = interval - (time.monotonic() - _OPENALEX_LAST_REQUEST)
        if remaining > 0:
            time.sleep(remaining)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        finally:
            _OPENALEX_LAST_REQUEST = time.monotonic()
    output: list[dict[str, Any]] = []
    for work in payload.get("results", []):
        if not isinstance(work, dict):
            continue
        openalex_id = str(work.get("id") or "")
        title = str(work.get("title") or work.get("display_name") or "").strip()
        if not openalex_id or not title:
            continue
        abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        primary = work.get("primary_location")
        best_oa = work.get("best_oa_location")
        open_access = work.get("open_access")
        primary = primary if isinstance(primary, dict) else {}
        best_oa = best_oa if isinstance(best_oa, dict) else {}
        open_access = open_access if isinstance(open_access, dict) else {}
        doi = str(work.get("doi") or "")
        landing_url = str(
            best_oa.get("landing_page_url")
            or primary.get("landing_page_url")
            or open_access.get("oa_url")
            or doi
            or openalex_id
        )
        authors = []
        for authorship in work.get("authorships", []) or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(str(author["display_name"]))
            if len(authors) >= 12:
                break
        source_id = "oa_" + openalex_id.rsplit("/", 1)[-1]
        output.append(
            {
                "source_id": source_id,
                "question": "",
                "query": query,
                "title": title[:500],
                "url": landing_url,
                "snippet": abstract[:2000],
                "retrieved_text": abstract[:20_000],
                "fetch_error": "",
                "source_kind": "scholarly_index_record",
                "provider": "openalex",
                "openalex_id": openalex_id,
                "doi": doi,
                "publication_year": work.get("publication_year"),
                "publication_date": work.get("publication_date"),
                "cited_by_count": work.get("cited_by_count"),
                "work_type": work.get("type"),
                "authors": authors,
                "code_retrieved": False,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    return output


class ResearchRetriever:
    """Run several precise searches and preserve every acceptance decision."""

    def __init__(
        self,
        task_name: str,
        council_dir: Path,
        *,
        forbidden_terms: Iterable[str] = (),
    ) -> None:
        self.task_name = str(task_name)
        self.council_dir = Path(council_dir)
        self.forbidden_terms = tuple(str(term) for term in forbidden_terms)
        self._audit: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _audit_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._audit.append(record)

    def _one_query(self, query: str, question: str) -> dict[str, Any]:
        valid, reason = validate_research_query(
            query,
            task_name=self.task_name,
            forbidden_terms=self.forbidden_terms,
        )
        self._audit_record(
            {"query": query, "question": question, "accepted": valid, "reason": reason}
        )
        if not valid:
            return {"sources": [], "retrieval_succeeded": True}
        if os.getenv("OPENALEX_API_KEY", "").strip():
            try:
                openalex_sources = _openalex_search(query)
                for source in openalex_sources:
                    source["question"] = question
                    self._audit_record(
                        {
                            "query": query,
                            "question": question,
                            "provider": "openalex",
                            "url": source.get("url"),
                            "title": source.get("title"),
                            "accepted": True,
                            "reason": "structured scholarly record returned by OpenAlex",
                        }
                    )
                if openalex_sources:
                    return {
                        "sources": openalex_sources,
                        "retrieval_succeeded": True,
                    }
                self._audit_record(
                    {
                        "query": query,
                        "question": question,
                        "provider": "openalex",
                        "accepted": False,
                        "reason": "OpenAlex returned no works; trying web fallbacks",
                    }
                )
            except Exception as exc:
                api_key = os.getenv("OPENALEX_API_KEY", "")
                safe_error = str(exc).replace(api_key, "<redacted>")[:500]
                self._audit_record(
                    {
                        "query": query,
                        "question": question,
                        "provider": "openalex",
                        "accepted": False,
                        "reason": f"OpenAlex failed; trying web fallbacks: {type(exc).__name__}: {safe_error}",
                    }
                )
        try:
            try:
                retry_budget = int(
                    os.getenv("AIBUILDAI_COUNCIL_SEARCH_RETRIES", "1")
                )
            except ValueError:
                retry_budget = 1
            raw = search_web(query, max_retries=max(1, min(3, retry_budget)))
        except Exception as exc:
            self._audit_record(
                {
                    "query": query,
                    "question": question,
                    "accepted": False,
                    "reason": f"search failed: {type(exc).__name__}: {exc}",
                }
            )
            return {"sources": [], "retrieval_succeeded": False}
        if not raw:
            self._audit_record(
                {
                    "query": query,
                    "question": question,
                    "accepted": False,
                    "reason": "all configured search strategies returned no results",
                }
            )
            return {"sources": [], "retrieval_succeeded": False}
        output: list[dict[str, Any]] = []
        for item in _parse_formatted_results(raw):
            url = item["url"]
            accepted, source_reason = _host_allowed(url)
            title_and_url = f"{item.get('title', '')} {url}".casefold()
            protected_result_terms = _task_tokens(self.task_name) | {
                term.strip().casefold()
                for term in self.forbidden_terms
                if len(term.strip()) >= 4
            }
            identity_hits = sorted(
                token for token in protected_result_terms if token in title_and_url
            )
            if identity_hits:
                accepted = False
                source_reason = "result reveals task identity: " + ", ".join(identity_hits)
            self._audit_record(
                {
                    "query": query,
                    "question": question,
                    "url": url,
                    "title": item.get("title", ""),
                    "accepted": accepted,
                    "reason": source_reason,
                }
            )
            if not accepted:
                continue
            text = ""
            fetch_error = ""
            try:
                text = _fetch_visible_text(url)
            except Exception as exc:
                fetch_error = f"{type(exc).__name__}: {exc}"[:500]
            source_id = "src_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
            output.append(
                {
                    "source_id": source_id,
                    "question": question,
                    "query": query,
                    "title": item.get("title", "")[:500],
                    "url": url,
                    "snippet": item.get("snippet", "")[:2000],
                    "retrieved_text": text[:20_000],
                    "fetch_error": fetch_error,
                    "source_kind": "primary_or_authoritative",
                    "code_retrieved": False,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        return {"sources": output, "retrieval_succeeded": True}

    def collect(self, requests: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        query_groups: list[tuple[str, list[str]]] = []
        for request in requests:
            question = " ".join(str(request.get("question") or "").split())
            queries = [
                " ".join(str(query).split())
                for query in request.get("queries", [])
                if str(query).strip()
            ]
            if queries:
                query_groups.append((question, queries))
        try:
            max_queries = max(
                1, min(32, int(os.getenv("AIBUILDAI_COUNCIL_MAX_QUERIES", "12")))
            )
        except ValueError:
            max_queries = 12
        # Round-robin selection preserves coverage across research questions
        # before spending quota on a second formulation of the same question.
        jobs: list[tuple[str, str]] = []
        seen_queries: set[str] = set()
        max_group_queries = max((len(queries) for _, queries in query_groups), default=0)
        for query_index in range(max_group_queries):
            for question, queries in query_groups:
                if query_index >= len(queries):
                    continue
                query = queries[query_index]
                key = query.casefold()
                if key in seen_queries:
                    self._audit_record(
                        {
                            "query": query,
                            "question": question,
                            "accepted": False,
                            "reason": "duplicate query removed before retrieval",
                        }
                    )
                    continue
                seen_queries.add(key)
                jobs.append((query, question))
                if len(jobs) >= max_queries:
                    break
            if len(jobs) >= max_queries:
                break
        unscheduled_seen = set(seen_queries)
        for question, queries in query_groups:
            for query in queries:
                key = query.casefold()
                if key in unscheduled_seen:
                    continue
                unscheduled_seen.add(key)
                self._audit_record(
                    {
                        "query": query,
                        "question": question,
                        "accepted": False,
                        "reason": f"not scheduled: council query budget is {max_queries}",
                    }
                )

        try:
            worker_count = max(
                1, min(3, int(os.getenv("AIBUILDAI_COUNCIL_SEARCH_WORKERS", "2")))
            )
        except ValueError:
            worker_count = 2
        try:
            failure_limit = max(
                1, int(os.getenv("AIBUILDAI_COUNCIL_SEARCH_FAILURE_LIMIT", "4"))
            )
        except ValueError:
            failure_limit = 4
        try:
            source_target = max(
                1, int(os.getenv("AIBUILDAI_COUNCIL_SOURCE_TARGET", "12"))
            )
        except ValueError:
            source_target = 12
        try:
            wave_delay = max(
                0.0, float(os.getenv("AIBUILDAI_COUNCIL_SEARCH_DELAY_SECONDS", "0.75"))
            )
        except ValueError:
            wave_delay = 0.75

        sources: list[dict[str, Any]] = []
        consecutive_failures = 0
        stop_reason = ""
        processed = 0
        for wave_start in range(0, len(jobs), worker_count):
            wave = jobs[wave_start : wave_start + worker_count]
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {
                    pool.submit(self._one_query, query, question): (query, question)
                    for query, question in wave
                }
                for future in as_completed(futures):
                    processed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"sources": [], "retrieval_succeeded": False}
                        self._audit_record(
                            {
                                "query": futures[future][0],
                                "question": futures[future][1],
                                "accepted": False,
                                "reason": f"research worker failed: {exc}",
                            }
                        )
                    sources.extend(list(result.get("sources", [])))
                    if result.get("retrieval_succeeded"):
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
            unique_count = len(
                {str(source.get("source_id")) for source in sources}
            )
            if unique_count >= source_target:
                stop_reason = f"primary-source evidence target reached ({unique_count})"
            elif consecutive_failures >= failure_limit:
                stop_reason = (
                    "search circuit breaker opened after "
                    f"{consecutive_failures} consecutive provider failures"
                )
            if stop_reason:
                break
            if wave_delay and wave_start + worker_count < len(jobs):
                time.sleep(wave_delay)
        if processed < len(jobs):
            for query, question in jobs[processed:]:
                self._audit_record(
                    {
                        "query": query,
                        "question": question,
                        "accepted": False,
                        "reason": stop_reason or "query budget exhausted",
                    }
                )
        unique: dict[str, dict[str, Any]] = {}
        for source in sources:
            unique.setdefault(str(source["source_id"]), source)
        ordered_sources = sorted(unique.values(), key=lambda item: str(item["source_id"]))
        audit = list(self._audit)
        self.council_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.council_dir / "research_sources.jsonl"
        source_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in ordered_sources),
            encoding="utf-8",
        )
        audit_path = self.council_dir / "query_audit.jsonl"
        audit_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in audit),
            encoding="utf-8",
        )
        return ordered_sources, audit
