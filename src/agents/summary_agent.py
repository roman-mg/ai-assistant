"""Summary Agent for summarizing search results."""

import traceback
from typing import TypedDict

from loguru import logger

from ..models.schemas import Paper, ResearchResult
from ..services.llm_service import llm_service


class SummaryState(TypedDict):
    """State for summary agent."""

    papers: list[Paper]
    web_results: list[dict]
    academic_results: list[dict]
    query: str
    summary: str
    research_result: ResearchResult | None
    error: str | None


class SummaryAgent:
    """Agent responsible for summarizing search results."""

    def __init__(self):
        """Initialize the summary agent."""
        self.name = "summary_agent"
        logger.info("Summary Agent initialized")

    async def summarize_papers(self, papers: list[Paper], query: str) -> str:
        """
        Summarize a list of research papers.
        
        Args:
            papers: List of papers to summarize
            query: Original research query
            
        Returns:
            Summary string
        """
        try:
            if not papers:
                return "No papers found to summarize."

            logger.info(f"Summarizing {len(papers)} papers for query: {query}")

            # Create summary prompt
            summary_prompt = self._create_paper_summary_prompt(papers, query)
            
            # Get summary from LLM
            summary = await llm_service.ainvoke_chat(summary_prompt)
            
            logger.info("Successfully generated paper summary")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing papers: {traceback.format_exc()}")
            return f"Error generating summary: {str(e)}"

    async def summarize_web_results(self, web_results: list[dict], query: str) -> str:
        """
        Summarize web search results.
        
        Args:
            web_results: List of web search results
            query: Original research query
            
        Returns:
            Summary string
        """
        try:
            if not web_results:
                return "No web results found to summarize."

            logger.info(f"Summarizing {len(web_results)} web results for query: {query}")

            # Create web summary prompt
            web_summary_prompt = self._create_web_summary_prompt(web_results, query)
            
            # Get summary from LLM
            summary = await llm_service.ainvoke_chat(web_summary_prompt)
            
            logger.info("Successfully generated web results summary")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing web results: {traceback.format_exc()}")
            return f"Error generating web summary: {str(e)}"

    async def create_comprehensive_summary(
        self, 
        papers: list[Paper], 
        web_results: list[dict], 
        academic_results: list[dict], 
        query: str
    ) -> str:
        """
        Create a comprehensive summary of all search results.
        
        Args:
            papers: List of research papers
            web_results: List of web search results
            academic_results: List of academic search results
            query: Original research query
            
        Returns:
            Comprehensive summary string
        """
        try:
            logger.info(f"Creating comprehensive summary for query: {query}")

            # Create comprehensive summary prompt
            comprehensive_prompt = self._create_comprehensive_summary_prompt(
                papers, web_results, academic_results, query
            )
            
            # Get comprehensive summary from LLM
            summary = await llm_service.ainvoke_chat(comprehensive_prompt)
            
            logger.info("Successfully generated comprehensive summary")
            return summary

        except Exception as e:
            logger.error(f"Error creating comprehensive summary: {traceback.format_exc()}")
            return f"Error generating comprehensive summary: {str(e)}"

    @staticmethod
    def _create_paper_summary_prompt(papers: list[Paper], query: str) -> str:
        """Create prompt for paper summarization."""
        papers_info = []
        for i, paper in enumerate(papers[:10], 1):  # Limit to top 10 papers
            papers_info.append(f"""
            Paper {i}:
            Title: {paper.title}
            Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
            Abstract: {paper.abstract[:500]}{'...' if len(paper.abstract) > 500 else ''}
            """)

        return f"""
        Please provide a comprehensive summary of the following research papers related to the query: "{query}"
        
        Papers found:
        {"".join(papers_info)}
        
        Please create a summary that:
        1. Identifies the main research themes and trends
        2. Highlights the most important findings and contributions
        3. Discusses methodologies and approaches used
        4. Identifies gaps or areas for future research
        5. Provides insights relevant to the original query
        
        Keep the summary concise but comprehensive, focusing on the most important insights.
        """

    @staticmethod
    def _create_web_summary_prompt(web_results: list[dict], query: str) -> str:
        """Create prompt for web results summarization."""
        web_info = []
        for i, result in enumerate(web_results[:5], 1):  # Limit to top 5 results
            title = result.get('title', 'No title')
            snippet = result.get('snippet', result.get('description', 'No description'))
            web_info.append(f"""
            Result {i}:
            Title: {title}
            Content: {snippet[:300]}{'...' if len(snippet) > 300 else ''}
            """)

        return f"""
        Please provide a summary of the following web search results related to the query: "{query}"
        
        Web results:
        {"".join(web_info)}
        
        Please create a summary that:
        1. Identifies key information and insights
        2. Highlights relevant trends or developments
        3. Provides context for the research query
        4. Identifies any additional resources or perspectives
        
        Keep the summary concise and focused on relevance to the research query.
        """

    @staticmethod
    def _create_comprehensive_summary_prompt(
        papers: list[Paper], 
        web_results: list[dict], 
        academic_results: list[dict], 
        query: str
    ) -> str:
        """Create prompt for comprehensive summarization."""
        
        # Paper summaries
        papers_summary = ""
        if papers:
            papers_summary = f"\n\nResearch Papers Found ({len(papers)} papers):\n"
            for i, paper in enumerate(papers[:5], 1):
                papers_summary += f"{i}. {paper.title} - {', '.join(paper.authors[:2])}\n"
        
        # Web results summary
        web_summary = ""
        if web_results:
            web_summary = f"\n\nWeb Search Results ({len(web_results)} results):\n"
            for i, result in enumerate(web_results[:3], 1):
                title = result.get('title', 'No title')
                web_summary += f"{i}. {title}\n"
        
        # Academic results summary
        academic_summary = ""
        if academic_results:
            academic_summary = f"\n\nAcademic Search Results ({len(academic_results)} results):\n"
            for i, result in enumerate(academic_results[:3], 1):
                title = result.get('title', 'No title')
                academic_summary += f"{i}. {title}\n"

        return f"""
        Please create a comprehensive research summary for the query: "{query}"
        
        {papers_summary}
        {web_summary}
        {academic_summary}
        
        Please provide a comprehensive summary that:
        1. Synthesizes findings from all sources
        2. Identifies key research themes and trends
        3. Highlights important contributions and methodologies
        4. Discusses implications and future directions
        5. Provides actionable insights relevant to the query
        
        Structure the summary with clear sections and make it informative and accessible.
        """

    async def create_research_result(
        self, 
        papers: list[Paper], 
        web_results: list[dict], 
        academic_results: list[dict], 
        query: str,
        search_time: float = 0.0
    ) -> ResearchResult:
        """
        Create a ResearchResult object with summary.
        
        Args:
            papers: List of research papers
            web_results: List of web search results
            academic_results: List of academic search results
            query: Original research query
            search_time: Time taken for search
            
        Returns:
            ResearchResult object
        """
        try:
            logger.info("Creating research result with summary")

            # Create comprehensive summary
            summary = await self.create_comprehensive_summary(
                papers, web_results, academic_results, query
            )

            # Determine sources used
            sources = ["arxiv"]
            if papers:  # If we have papers, vector store was likely used
                sources.append("vector_store")
            if web_results:
                sources.append("web_search")
            if academic_results:
                sources.append("academic_search")

            # Create research result
            research_result = ResearchResult(
                papers=papers,
                total_found=len(papers),
                search_query=query,
                search_time=search_time,
                sources=sources,
                summary=summary
            )

            logger.info("Successfully created research result")
            return research_result

        except Exception as e:
            logger.error(f"Error creating research result: {traceback.format_exc()}")
            return ResearchResult(
                papers=papers,
                total_found=len(papers),
                search_query=query,
                search_time=search_time,
                sources=["arxiv"],
                error=str(e),
            )

    async def process_state(self, state: SummaryState) -> SummaryState:
        """
        Process the summary state.
        
        Args:
            state: Current state containing search results
            
        Returns:
            Updated state with summary and research result
        """
        try:
            papers = state["papers"]
            web_results = state["web_results"]
            academic_results = state["academic_results"]
            query = state["query"]
            
            # Create comprehensive summary
            summary = await self.create_comprehensive_summary(
                papers, web_results, academic_results, query
            )
            
            # Create research result
            research_result = await self.create_research_result(
                papers, web_results, academic_results, query
            )
            
            # Update state
            state["summary"] = summary
            state["research_result"] = research_result
            state["error"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing summary state: {traceback.format_exc()}")
            state["error"] = str(e)
            state["summary"] = f"Error generating summary: {str(e)}"
            state["research_result"] = None
            return state


# Global summary agent instance
summary_agent = SummaryAgent()
