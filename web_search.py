"""
web_search.py
─────────────
Web search fallback for the agentic pipeline.
  Primary  : Tavily API  (set TAVILY_API_KEY env variable)
  Fallback : DuckDuckGo  (free, no key required)
"""
from __future__ import annotations
import os


class WebSearcher:

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return list of {title, url, snippet}. Never raises."""
        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if api_key:
            results = self._tavily(query, api_key, max_results)
            if results:
                return results
        return self._duckduckgo(query, max_results)

    def as_context(self, results: list[dict]) -> str:
        """Flatten results into a single LLM context string."""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Web Result {i}: {r['title']}]\n"
                f"Source: {r['url']}\n"
                f"{r['snippet']}"
            )
        return "\n\n".join(parts)

    # ── Tavily ─────────────────────────────────────────────────────────────────

    def _tavily(self, query: str, api_key: str, max_results: int) -> list[dict]:
        try:
            from tavily import TavilyClient
            client  = TavilyClient(api_key=api_key)
            resp    = client.search(query, max_results=max_results)
            results = []
            for r in resp.get("results", []):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "snippet": r.get("content", ""),
                })
            return results
        except ImportError:
            return []
        except Exception:
            return []

    # ── DuckDuckGo fallback ────────────────────────────────────────────────────

    def _duckduckgo(self, query: str, max_results: int) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title":   r.get("title", ""),
                        "url":     r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except ImportError:
            return [{"title": "No search engine available", "url": "",
                     "snippet": "pip install tavily-python  OR  pip install duckduckgo-search"}]
        except Exception as exc:
            return [{"title": "Search failed", "url": "", "snippet": str(exc)}]