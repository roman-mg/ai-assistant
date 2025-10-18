"""Search Agent for handling ArXiv, web, and academic search functionality."""

import traceback
from typing import TypedDict

from loguru import logger

from ..config.settings import settings
from ..models.schemas import Paper
from ..tools.arxiv_tool import ArxivTool, RecentPapersTool
from ..tools.web_search_tool import WebSearchTool
from ..vectorstore.faiss_store import vector_store


class SearchState(TypedDict):
    """State for search agent."""

    query: str
    papers: list[Paper]
    web_results: list[dict]
    error: str | None


class SearchAgent:
    """Agent responsible for searching papers from various sources."""

    def __init__(self):
        """Initialize the search agent."""
        self.name = "search_agent"
        
        # Initialize tools
        self.tools = {
            "arxiv": ArxivTool(),
            "recent_papers": RecentPapersTool(),
            "web_search": WebSearchTool(),
        }
        
        logger.info("Search Agent initialized")

    async def search_papers(self, query: str) -> list[Paper]:
        """
        Search for papers using ArXiv.
        
        Args:
            query: The research query
            
        Returns:
            List of found papers
        """
        try:
            logger.info(f"Searching papers for: {query}")
            
            # Use ArXiv tool to search for papers
            papers = await self.tools["arxiv"]._arun(
                query=query,
                max_results=settings.researcher.max_papers_per_query,
            )
            
            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers

        except Exception as e:
            logger.error(f"Error searching papers: {traceback.format_exc()}")
            return []

    async def search_recent_papers(self, query: str, categories: list[str] = None) -> list[Paper]:
        """
        Search for recent papers in specific categories.
        
        Args:
            query: The research query
            categories: List of ArXiv categories to search
            
        Returns:
            List of recent papers
        """
        try:
            if categories is None:
                categories = ["cs.AI", "cs.LG", "cs.CV"]
            
            logger.info(f"Searching recent papers in categories: {categories}")
            
            all_papers = []
            papers_per_category = settings.researcher.max_papers_per_query // len(categories)
            
            for category in categories:
                try:
                    papers = await self.tools["recent_papers"]._arun(
                        category=category,
                        days_back=7,
                        max_results=papers_per_category
                    )
                    all_papers.extend(papers)
                except Exception as e:
                    logger.warning(f"Error searching recent papers in {category}: {e}")
                    continue
            
            logger.info(f"Found {len(all_papers)} recent papers")
            return all_papers[:settings.researcher.max_papers_per_query]
            
        except Exception:
            logger.error(f"Error searching recent papers: {traceback.format_exc()}")
            return []

    @staticmethod
    async def search_vector_store(query: str, existing_papers: list[Paper] = None) -> list[Paper]:
        """
        Search for similar papers in the vector store.
        
        Args:
            query: The research query
            existing_papers: Papers already found to avoid duplicates
            
        Returns:
            List of additional papers from vector store
        """
        try:
            logger.info(f"Searching vector store for: {query}")

            # Search vector store for similar papers
            similar_papers = vector_store.search_similar_papers(
                query=query,
                k=5,  # Get top 5 similar papers
                similarity_threshold=settings.vector_store.similarity_threshold,
            )

            # Add similar papers to existing papers (avoid duplicates)
            if existing_papers is None:
                existing_papers = []
            
            existing_titles = {paper.title.lower() for paper in existing_papers}
            additional_papers = []
            
            for paper, score in similar_papers:
                if paper.title.lower() not in existing_titles:
                    # Add similarity score to paper
                    paper.similarity_score = score
                    additional_papers.append(paper)

            logger.info(f"Found {len(additional_papers)} additional papers in vector store")
            return additional_papers

        except Exception:
            logger.error(f"Error searching vector store: {traceback.format_exc()}")
            return []

    async def web_search(self, query: str) -> list[dict]:
        """
        Perform web search for additional context.
        
        Args:
            query: The research query
            
        Returns:
            List of web search results
        """
        try:
            if not settings.web_search.enabled:
                logger.info("Web search disabled, skipping")
                return []

            logger.info(f"Performing web search for: {query}")

            # Use web search tool
            web_results = await self.tools["web_search"]._arun(
                query=query,
                max_results=settings.web_search.max_results,
            )

            logger.info(f"Found {len(web_results)} web search results")
            return web_results

        except Exception:
            logger.error(f"Error in web search: {traceback.format_exc()}")
            return []

    async def comprehensive_search(self, query: str) -> tuple[list[Paper], list[dict]]:
        """
        Perform comprehensive search across all sources.
        
        Args:
            query: The research query
            
        Returns:
            Tuple of (papers, web_results, academic_results)
        """
        try:
            logger.info(f"Performing comprehensive search for: {query}")
            
            # Search papers from ArXiv
            papers = await self.search_papers(query)
            
            # Search vector store for additional papers
            additional_papers = await self.search_vector_store(query, papers)
            papers.extend(additional_papers)
            
            # Perform web search
            web_results = await self.web_search(query)
            
            logger.info(f"Comprehensive search completed: {len(papers)} papers, {len(web_results)} web results.")
            
            return papers, web_results
            
        except Exception:
            logger.error(f"Error in comprehensive search: {traceback.format_exc()}")
            return [], []

    async def process_state(self, state: SearchState) -> SearchState:
        """
        Process the search state.
        
        Args:
            state: Current state containing query
            
        Returns:
            Updated state with search results
        """
        try:
            query = state["query"]
            
            # Perform comprehensive search
            papers, web_results = await self.comprehensive_search(query)
            
            # Update state
            state["papers"] = papers
            state["web_results"] = web_results
            state["error"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing search state: {traceback.format_exc()}")
            state["error"] = str(e)
            state["papers"] = []
            state["web_results"] = []
            return state


# Global search agent instance
search_agent = SearchAgent()
