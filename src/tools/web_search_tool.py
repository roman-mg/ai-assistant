"""Web search tool for additional research capabilities."""

import requests  # TODO: use aiohttp
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from ..config.settings import settings


class WebSearchInput(BaseModel):
    """Input for web search tool."""

    query: str = Field(..., description="Search query for web search")
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1,
        le=20,
    )


class WebSearchTool(BaseTool):
    """Tool for web search using DuckDuckGo."""

    name: str = "web_search"
    description: str = (
        "Search the web for additional information about research topics. "
        "Use this tool to find recent news, blog posts, or additional resources "
        "related to AI/ML research."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5
    ) -> list[dict]:
        """Perform web search and return results."""
        if not settings.web_search.enabled:
            logger.info("Web search is disabled")
            return []

        try:
            logger.info(f"Performing web search for: {query}")

            match settings.web_search.search_engine:
                case "duckduckgo":
                    results = self._search_duckduckgo(query, max_results)
                case _:
                    logger.warning(f"Unsupported search engine: {settings.web_search.search_engine}")
                    return []

            logger.info(f"Found {len(results)} web search results")
            return results

        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []

    @staticmethod
    def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
        """Search using DuckDuckGo."""
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            # Extract instant answer
            if data.get("Abstract"):
                results.append(
                    {
                        "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", ""),
                        "source": "DuckDuckGo Instant Answer",
                    }
                )

            # Extract related topics
            for topic in data.get("RelatedTopics", [])[: max_results - len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(
                        {
                            "title": topic.get("Text", "").split(" - ")[0],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                            "source": "DuckDuckGo Related Topics",
                        }
                    )

            return results[:max_results]

        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Async version of web search."""
        return self._run(query, max_results)


class AcademicSearchInput(BaseModel):
    """Input for academic web search."""

    query: str = Field(..., description="Academic search query")
    max_results: int = Field(
        default=5,
        description="Maximum number of results",
        ge=1,
        le=10,
    )


class AcademicSearchTool(BaseTool):
    """Tool for searching academic websites and repositories."""

    name: str = "academic_search"
    description: str = (
        "Search academic websites like Google Scholar, ResearchGate, and arXiv "
        "for additional research papers and academic resources."
    )
    args_schema: type[BaseModel] = AcademicSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Search academic websites."""
        try:
            logger.info(f"Performing academic search for: {query}")

            results = []

            # Search Google Scholar (simplified)
            scholar_results = self._search_google_scholar(query, max_results // 2)
            results.extend(scholar_results)

            # Search ResearchGate (simplified)
            rg_results = self._search_researchgate(query, max_results - len(results))
            results.extend(rg_results)

            logger.info(f"Found {len(results)} academic search results")
            return results[:max_results]

        except Exception as e:
            logger.error(f"Error performing academic search: {e}")
            return []

    @staticmethod
    def _search_google_scholar(query: str, max_results: int) -> list[dict]:
        """Search Google Scholar (simplified implementation)."""
        try:
            # Note: This is a simplified implementation
            # In production, you might want to use a proper Google Scholar API
            url = "https://scholar.google.com/scholar"
            params = {"q": query, "hl": "en"}

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            results = []

            # Extract search results (simplified parsing)
            for result in soup.find_all("div", class_="gs_ri")[:max_results]:
                title_elem = result.find("h3", class_="gs_rt")
                if title_elem:
                    title = title_elem.get_text().strip()
                    link = title_elem.find("a")
                    url = link.get("href") if link else ""

                    snippet_elem = result.find("div", class_="gs_rs")
                    snippet = snippet_elem.get_text().strip() if snippet_elem else ""

                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source": "Google Scholar",
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []

    @staticmethod
    def _search_researchgate(query: str, max_results: int) -> list[dict]:
        """Search ResearchGate (simplified implementation)."""
        try:
            # Note: This is a simplified implementation
            # ResearchGate has strict anti-bot measures
            url = "https://www.researchgate.net/search"
            params = {"q": query}

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            results = []

            # Extract search results (simplified parsing)
            for result in soup.find_all("div", class_="nova-e-text")[:max_results]:
                title = result.get_text().strip()
                if title and len(title) > 10:  # Basic filtering
                    results.append(
                        {
                            "title": title,
                            "url": "",
                            "snippet": title,
                            "source": "ResearchGate",
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error searching ResearchGate: {e}")
            return []

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """Async version of academic search."""
        return self._run(query, max_results)
