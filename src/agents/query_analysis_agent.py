"""Query Analysis Agent for analyzing and optimizing research queries."""

import traceback
from typing import TypedDict

from loguru import logger

from ..services.llm_service import llm_service


class QueryAnalysisState(TypedDict):
    """State for query analysis agent."""

    original_query: str
    analyzed_query: str
    error: str | None


class QueryAnalysisAgent:
    """Agent responsible for analyzing and optimizing research queries."""

    def __init__(self):
        """Initialize the query analysis agent."""
        self.name = "query_analysis_agent"
        logger.info("Query Analysis Agent initialized")

    async def analyze_query(self, query: str) -> str:
        """
        Analyze the research query and return an optimized query string.
        
        Args:
            query: The original research query (already sanitized by security agent)
            
        Returns:
            Optimized query string for research
        """
        try:
            logger.info(f"Analyzing query: {query}")

            # Use LLM to analyze and optimize the query
            analysis_prompt = f"""
            SYSTEM:
            You are a research query optimizer. Your job is to analyze user queries and return 
            clean, research-focused query strings optimized for academic paper search.
            
            USER QUERY TO ANALYZE:
            {query}
            
            Please analyze this query and return a clean, research-focused query string that:
            - Focuses on academic research topics
            - Uses proper academic terminology
            - Removes any non-research related content
            - Is optimized for paper search engines like ArXiv
            - Maintains the core research intent
            
            Return ONLY the optimized query string, no explanations or additional text.
            """

            response = await llm_service.ainvoke_chat(analysis_prompt)
            
            # Clean the response to ensure it's just a query string
            cleaned_query = self._clean_response(response)
            
            logger.info(f"Query analysis completed: '{query}' -> '{cleaned_query}'")
            return cleaned_query

        except Exception:
            logger.error(f"Error analyzing query: {traceback.format_exc()}")
            # Return original query if analysis fails
            return query

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Clean the LLM response to extract just the query string.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned query string
        """
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "here is the optimized query:",
            "the optimized query is:",
            "query:",
            "research query:",
            "optimized query:",
            "cleaned query:",
        ]
        
        suffixes_to_remove = [
            "this query focuses on",
            "this query is optimized for",
            "the query has been optimized",
            "this query is designed for",
        ]
        
        cleaned = response.strip()
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove suffixes
        for suffix in suffixes_to_remove:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[:-len(suffix)].strip()
                break
        
        # Remove quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        
        # Ensure it's not empty
        if not cleaned.strip():
            return "artificial intelligence research"
        
        return cleaned.strip()

    async def process_state(self, state: QueryAnalysisState) -> QueryAnalysisState:
        """
        Process the query analysis state.
        
        Args:
            state: Current state containing original query
            
        Returns:
            Updated state with analyzed query
        """
        try:
            original_query = state["original_query"]
            analyzed_query = await self.analyze_query(original_query)
            
            state["analyzed_query"] = analyzed_query
            state["error"] = None
            
            return state
            
        except Exception as e:
            logger.error(f"Error processing query analysis state: {traceback.format_exc()}")
            state["error"] = str(e)
            state["analyzed_query"] = state["original_query"]  # Fallback to original
            return state


# Global query analysis agent instance
query_analysis_agent = QueryAnalysisAgent()
