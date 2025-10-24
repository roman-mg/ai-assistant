"""Query Analysis Agent for analyzing and optimizing research queries."""

import json
import traceback
from typing import Any, TypedDict

from loguru import logger

from ..services.llm_service import llm_service


class QueryAnalysisState(TypedDict):
    """State for query analysis agent."""

    original_query: str
    analyzed_query: str
    suggested_query: str | None
    analysis_data: dict[str, Any] | None
    requires_hitl: bool
    error: str | None


class QueryAnalysisAgent:
    """Agent responsible for analyzing and optimizing research queries."""

    def __init__(self):
        """Initialize the query analysis agent."""
        self.name = "query_analysis_agent"
        logger.info("Query Analysis Agent initialized with HITL enabled")

    async def analyze_query(self, query: str) -> tuple[str, dict[str, Any] | None]:
        """
        Summarize the user query and return it in JSON format.

        Args:
            query: The original research query (already sanitized by security agent)

        Returns:
            Tuple of (JSON string containing summarized query, analysis data dict)
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
            response_text = response.strip()

            # Parse JSON response to extract analysis data
            analysis_data = None
            try:
                analysis_data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse analysis response as JSON: {response_text[:100]}")

            # Return the JSON response directly
            logger.info(f"Query analysis completed: '{query}' -> JSON summary received")
            return response_text, analysis_data

        except Exception:
            logger.error(f"Error analyzing query: {traceback.format_exc()}")
            # Return original query if analysis fails
            return query, None

    @staticmethod
    async def _suggest_improved_query(query: str, analysis_data: dict[str, Any] | None = None) -> str:
        """
        Generate an improved version of the query for better research results.

        Args:
            query: The original research query
            analysis_data: Optional analysis metadata

        Returns:
            Suggested improved query
        """
        try:
            logger.info(f"Generating improved query suggestion for: {query}")

            context_info = ""
            if analysis_data:
                context_info = f"""
                Analysis Context:
                - Main Topic: {analysis_data.get("main_topic", "N/A")}
                - Focus Area: {analysis_data.get("focus_area", "N/A")}
                - Key Terms: {", ".join(analysis_data.get("key_terms", []))}
                """

            improvement_prompt = f"""
            SYSTEM:
            You are an expert at refining research queries to improve search results.
            Your job is to take a user's query and suggest an improved, more specific version
            that would yield better academic research results.
            
            ORIGINAL QUERY:
            {query}
            {context_info}
            
            Generate an improved version of this query that:
            - Is more specific and focused
            - Uses appropriate academic/technical terminology
            - Maintains the user's intent
            - Would yield better search results in academic databases
            
            Return ONLY the improved query text, without any explanations or additional text.
            """

            improved_query = await llm_service.ainvoke_chat(improvement_prompt)
            improved_query = improved_query.strip()

            logger.info(f"Generated improved query: '{improved_query}'")
            return improved_query

        except Exception:
            logger.error(f"Error generating improved query: {traceback.format_exc()}")
            # Return original query if improvement fails
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
            analyzed_query, analysis_data = await self.analyze_query(original_query)

            state["analyzed_query"] = analyzed_query
            state["analysis_data"] = analysis_data
            state["error"] = None

            # Generate improved query suggestion (HITL always enabled)
            suggested_query = await self._suggest_improved_query(original_query, analysis_data)
            state["suggested_query"] = suggested_query
            state["requires_hitl"] = True
            logger.info(f"HITL - suggested query: {suggested_query}")

            return state

        except Exception as e:
            logger.error(f"Error processing query analysis state: {traceback.format_exc()}")
            state["error"] = str(e)
            state["analyzed_query"] = state["original_query"]  # Fallback to original
            state["suggested_query"] = None
            state["analysis_data"] = None
            state["requires_hitl"] = False
            return state


# Global query analysis agent instance
query_analysis_agent = QueryAnalysisAgent()
