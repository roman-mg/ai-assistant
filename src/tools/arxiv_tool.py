"""ArXiv research tool for finding academic papers."""

import traceback
from datetime import datetime, timedelta

import arxiv
from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..models.schemas import Paper


class ArxivSearchInput(BaseModel):
    """Input for ArXiv search tool."""

    query: str = Field(..., description="Search query for ArXiv papers")
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort order: relevance, lastUpdatedDate, submittedDate",
    )
    sort_order: str = Field(
        default="descending",
        description="Sort order: ascending or descending",
    )


class ArxivTool(BaseTool):
    """Tool for searching ArXiv for academic papers."""

    name: str = "arxiv_search"
    description: str = (
        "Search ArXiv for academic papers. Use this tool to find recent research papers "
        "on specific topics. The query should be clear and specific about the research area."
    )
    args_schema: type[BaseModel] = ArxivSearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[Paper]:
        """Search ArXiv for papers matching the query."""
        try:
            logger.info(f"Searching ArXiv for: {query}")

            # Configure search parameters
            search_params = {
                "query": query,
                "max_results": min(max_results, settings.researcher.arxiv_max_results),
                "sort_by": getattr(arxiv.SortCriterion, sort_by, arxiv.SortCriterion.Relevance),
                "sort_order": getattr(arxiv.SortOrder, sort_order.title(), arxiv.SortOrder.Descending),
            }

            # Perform search
            search = arxiv.Search(**search_params)
            client = arxiv.Client()
            results = list(client.results(search))

            papers = []
            for result in results:
                try:
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        summary="",  # Will be filled by paper analyzer
                        url=result.entry_id,
                        published_date=result.published,
                        categories=result.categories,
                    )
                    papers.append(paper)
                except Exception:
                    logger.warning(f"Error processing paper {result.title}: {traceback.format_exc()}")
                    continue

            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers

        except Exception:
            logger.error(f"Error searching ArXiv: {traceback.format_exc()}")
            return []

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[Paper]:
        """Async version of ArXiv search."""
        return self._run(query, max_results, sort_by, sort_order)


class RecentPapersInput(BaseModel):
    """Input for recent papers search."""

    category: str = Field(
        default="cs.AI",
        description="ArXiv category (e.g., cs.AI, cs.LG, cs.CV)",
    )
    days_back: int = Field(
        default=7,
        description="Number of days back to search",
        ge=1,
        le=30,
    )
    max_results: int = Field(
        default=20,
        description="Maximum number of results",
        ge=1,
        le=100,
    )


class RecentPapersTool(BaseTool):
    """Tool for finding recent papers in a specific category."""

    name: str = "recent_papers"
    description: str = (
        "Find recent papers in a specific ArXiv category. Useful for staying up-to-date "
        "with the latest research in a particular field."
    )
    args_schema: type[BaseModel] = RecentPapersInput

    def _run(
        self,
        category: str = "cs.AI",
        days_back: int = 7,
        max_results: int = 20,
    ) -> list[Paper]:
        """Find recent papers in a specific category."""
        try:
            logger.info(f"Searching for recent papers in {category} from last {days_back} days")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Create date range query
            date_query = (
                f"cat:{category} AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
            )

            # Search with date filter
            arxiv.Client()
            search = arxiv.Search(
                query=date_query,
                max_results=min(max_results, settings.researcher.arxiv_max_results),
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            client = arxiv.Client()
            results = list(client.results(search))

            papers = []
            for result in results:
                try:
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        abstract=result.summary,
                        summary="",  # Will be filled by paper analyzer
                        url=result.entry_id,
                        published_date=result.published,
                        categories=result.categories,
                    )
                    papers.append(paper)
                except Exception:
                    logger.warning(f"Error processing paper {result.title}: {traceback.format_exc()}")
                    continue

            logger.info(f"Found {len(papers)} recent papers in {category}")
            return papers

        except Exception:
            logger.error(f"Error searching recent papers: {traceback.format_exc()}")
            return []

    async def _arun(
        self,
        category: str = "cs.AI",
        days_back: int = 7,
        max_results: int = 20,
    ) -> list[Paper]:
        """Async version of recent papers search."""
        return self._run(category, days_back, max_results)
