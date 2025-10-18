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
        Summarize the user query and return it in JSON format.
        
        Args:
            query: The original research query (already sanitized by security agent)
            
        Returns:
            JSON string containing summarized query
        """
        try:
            logger.info(f"Analyzing query: {query}")

            # Use LLM to summarize the query and format as JSON
            analysis_prompt = f"""
            SYSTEM:
            You are a query summarizer. Your job is to analyze user queries and provide 
            a clear, concise summary in JSON format.
            
            USER QUERY TO SUMMARIZE:
            {query}
            
            Please analyze this query and provide a JSON object that summarizes:
            - The main topic or subject
            - The specific aspect or focus area
            - Any key terms or concepts mentioned
            
            Return ONLY a JSON object with this structure:
            {{
                "main_topic": "primary subject area",
                "focus_area": "specific aspect being asked about", 
                "key_terms": ["term1", "term2", "term3"],
                "query_summary": "concise summary of what the user is asking"
            }}
            
            Do not include any other text, explanations, or additional data.
            """

            response = await llm_service.ainvoke_chat(analysis_prompt)
            
            # Return the JSON response directly
            logger.info(f"Query analysis completed: '{query}' -> JSON summary received")
            return response.strip()

        except Exception:
            logger.error(f"Error analyzing query: {traceback.format_exc()}")
            # Return original query if analysis fails
            return query

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
