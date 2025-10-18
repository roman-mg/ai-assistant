"""Web search tool using MCP (Model Context Protocol) and Wikipedia API."""

import asyncio
import traceback

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
    academic_focus: bool = Field(
        default=False,
        description="Whether to focus on academic content"
    )


class WebSearchTool(BaseTool):
    """Tool for web search using MCP."""

    name: str = "web_search"
    description: str = (
        "Search Wikipedia for information about research topics using MCP. "
        "Use this tool to find comprehensive articles, definitions, and resources "
        "related to AI/ML research. Set academic_focus=True for academic content."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def __init__(self):
        """Initialize the web search tool."""
        super().__init__()

    def _run(
        self,
        query: str,
        max_results: int = 5,
        academic_focus: bool = False
    ) -> list[dict]:
        """Perform web search and return results (sync wrapper)."""
        return asyncio.run(self._arun(query, max_results, academic_focus))

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        academic_focus: bool = False
    ) -> list[dict]:
        """Perform web search using MCP only."""
        if not settings.web_search.enabled:
            logger.info("Web search is disabled")
            return []

        try:
            search_type = "academic" if academic_focus else "general"
            logger.info(f"Performing MCP {search_type} Wikipedia search for: {query}")

            # Use MCP for web search only
            results = await self._search_via_mcp(query, max_results, academic_focus)

            logger.info(f"Found {len(results)} Wikipedia search results")
            return results

        except Exception:
            logger.error(f"Error performing MCP Wikipedia search: {traceback.format_exc()}")
            return []

    @staticmethod
    async def _search_via_mcp(query: str, max_results: int, academic_focus: bool = False) -> list[dict]:
        """Search using MCP (Model Context Protocol) only."""
        try:
            search_type = "academic" if academic_focus else "general"
            logger.info(f"Using MCP for {search_type} Wikipedia search")
            
            # MCP Wikipedia search implementation
            method = "academic_wikipedia_search" if academic_focus else "wikipedia_search"
            mcp_request = {
                "method": method,
                "params": {
                    "query": query,
                    "max_results": max_results,
                    "language": "en"
                }
            }
            
            # Add academic focus parameter if needed
            if academic_focus:
                mcp_request["params"]["focus"] = "academic"
            
            logger.info(f"MCP request: {mcp_request}")
            
            # TODO: Implement actual MCP client call
            # This should call a real MCP server that handles Wikipedia search
            logger.warning("MCP client not implemented - returning empty results")
            return []
            
        except Exception:
            logger.error(f"Error in MCP Wikipedia search: {traceback.format_exc()}")
            return []
