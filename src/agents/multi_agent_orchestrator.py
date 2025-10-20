"""Multi-Agent Orchestrator for coordinating the 3 specialized agents."""

import time
import traceback
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from ..config.settings import settings
from ..models.schemas import ResearchResult
from ..vectorstore.faiss_store import vector_store
from .query_analysis_agent import QueryAnalysisState, query_analysis_agent
from .search_agent import SearchState, search_agent
from .security_agent import SecurityState, security_agent
from .summary_agent import SummaryState, summary_agent


class MultiAgentState(TypedDict):
    """State for the multi-agent research system."""

    # Input
    research_query: str
    conversation_history: list[dict[str, Any]]

    # Security Analysis
    original_input: str
    sanitized_input: str
    is_safe: bool
    threat_level: str
    detected_threats: list[str]

    # Query Analysis
    original_query: str
    analyzed_query: str

    # Search Results
    papers: list
    web_results: list[dict]
    academic_results: list[dict]

    # Summary Results
    summary: str
    research_result: ResearchResult | None

    # Control
    current_step: str
    error: str | None


class MultiAgentOrchestrator:
    """Orchestrator for coordinating query analysis, search, and summary agents."""

    def __init__(self):
        """Initialize the multi-agent orchestrator."""
        self.name = "multi_agent_orchestrator"

        # Initialize agents
        self.agents = {
            "security": security_agent,
            "query_analysis": query_analysis_agent,
            "search": search_agent,
            "summary": summary_agent,
        }

        # Build the graph
        self.graph = self._build_graph()

        # Save graph diagram
        try:
            with open(settings.researcher.graph_diagram_path, "wb") as f:
                f.write(self.graph.get_graph(xray=1).draw_mermaid_png())
        except Exception as e:
            logger.warning(f"Could not save graph diagram: {e}")

        logger.info("Multi-Agent Orchestrator initialized")

    async def research(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] = None,
    ) -> ResearchResult:
        """
        Perform research using the multi-agent system.

        Args:
            query: Research query
            conversation_history: Previous conversation context

        Returns:
            Research result with papers and summary
        """
        logger.info(f"Starting multi-agent research for query: {query}")
        start_time = time.time()

        try:
            # Initialize state
            initial_state: MultiAgentState = {
                "research_query": query,
                "conversation_history": conversation_history or [],
                "original_input": query,
                "sanitized_input": "",
                "is_safe": True,
                "threat_level": "none",
                "detected_threats": [],
                "original_query": query,
                "analyzed_query": "",
                "papers": [],
                "web_results": [],
                "academic_results": [],
                "summary": "",
                "research_result": None,
                "current_step": "security",
                "error": None,
            }

            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Calculate search time
            search_time = time.time() - start_time

            # Get research result
            if final_state.get("research_result"):
                result = final_state["research_result"]
                result.search_time = search_time
            else:
                # Fallback: create basic result
                result = ResearchResult(
                    papers=final_state.get("papers", []),
                    total_found=len(final_state.get("papers", [])),
                    search_query=query,
                    search_time=search_time,
                    sources=self._determine_sources(final_state),
                )

            logger.info(f"Multi-agent research completed in {search_time:.2f}s, found {len(result.papers)} papers")
            return result

        except Exception as e:
            logger.error(f"Error in multi-agent research: {traceback.format_exc()}")
            return ResearchResult(
                papers=[],
                total_found=0,
                search_query=query,
                search_time=time.time() - start_time,
                sources=[],
                error=str(e),
            )

    @staticmethod
    def _determine_sources(state: MultiAgentState) -> list[str]:
        """Determine which sources were used based on state."""
        sources = ["arxiv"]

        if state.get("papers"):
            sources.append("vector_store")

        if state.get("web_results"):
            sources.append("web_search")

        if state.get("academic_results"):
            sources.append("academic_search")

        return sources

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for multi-agent coordination."""
        builder = StateGraph(MultiAgentState)

        # Add nodes for each agent
        builder.add_node("security", self._security_node)
        builder.add_node("query_analysis", self._query_analysis_node)
        builder.add_node("search", self._search_node)
        builder.add_node("summary", self._summary_node)

        # Add edges
        builder.set_entry_point("security")

        # Conditional routing based on security analysis
        builder.add_conditional_edges(
            "security",
            self._should_continue_after_security,
            {"continue": "query_analysis", "skip_to_summary": "summary"},
        )

        builder.add_edge("query_analysis", "search")
        builder.add_edge("search", "summary")
        builder.add_edge("summary", END)

        return builder.compile()

    @staticmethod
    def _should_continue_after_security(state: MultiAgentState) -> str:
        """Determine whether to continue with query analysis or skip to summary."""
        if not state.get("is_safe", True):
            logger.info("Skipping query analysis due to security threat")
            decision = "skip_to_summary"
        else:
            logger.info("Continuing with query analysis after security check")
            decision = "continue"

        return decision

    async def _security_node(self, state: MultiAgentState) -> MultiAgentState:
        """Security analysis node."""
        try:
            logger.info("Executing security node")

            security_state: SecurityState = {
                "original_input": state["research_query"],
                "sanitized_input": "",
                "is_safe": True,
                "threat_level": "none",
                "detected_threats": [],
                "error": None,
            }

            result_state = await self.agents["security"].process_state(security_state)

            # Update state with security results
            state["original_input"] = result_state["original_input"]
            state["sanitized_input"] = result_state["sanitized_input"]
            state["is_safe"] = result_state["is_safe"]
            state["threat_level"] = result_state["threat_level"]
            state["detected_threats"] = result_state["detected_threats"]
            state["current_step"] = "query_analysis"
            state["error"] = result_state.get("error")

            # Log security analysis results
            if not state["is_safe"]:
                logger.warning(f"Security threat detected: {state['threat_level']} - {state['detected_threats']}")
            else:
                logger.info("Security analysis passed - input is safe")

            return state

        except Exception as e:
            logger.error(f"Error in security node: {traceback.format_exc()}")
            state["error"] = str(e)
            state["is_safe"] = False
            state["threat_level"] = "critical"
            state["sanitized_input"] = "artificial intelligence research"
            return state

    async def _query_analysis_node(self, state: MultiAgentState) -> MultiAgentState:
        """Query analysis node."""
        try:
            logger.info("Executing query analysis node")

            # Use sanitized input from security analysis
            input_query = state["sanitized_input"] if state["sanitized_input"] else state["research_query"]

            query_state: QueryAnalysisState = {
                "original_query": input_query,
                "analyzed_query": "",
                "error": None,
            }

            result_state = await self.agents["query_analysis"].process_state(query_state)

            state["original_query"] = result_state["original_query"]
            state["analyzed_query"] = result_state["analyzed_query"]
            state["current_step"] = "search"
            state["error"] = result_state.get("error")

            logger.info(f"Query analysis completed: '{state['original_query']}' -> '{state['analyzed_query']}'")

            return state

        except Exception as e:
            logger.error(f"Error in query analysis node: {traceback.format_exc()}")
            state["error"] = str(e)
            state["analyzed_query"] = state["research_query"]  # Fallback to original
            return state

    async def _search_node(self, state: MultiAgentState) -> MultiAgentState:
        """Search node."""
        try:
            logger.info("Executing search node")

            # Use analyzed query for search
            search_query = state["analyzed_query"] or state["research_query"]

            search_state: SearchState = {
                "query": search_query,
                "papers": [],
                "web_results": [],
                "error": None,
            }

            result_state = await self.agents["search"].process_state(search_state)

            state["papers"] = result_state["papers"]
            state["web_results"] = result_state["web_results"]
            state["academic_results"] = result_state["academic_results"]
            state["current_step"] = "summary"
            state["error"] = result_state.get("error")

            logger.info(
                f"Search completed: {len(state['papers'])} papers, {len(state['web_results'])} web results, {len(state['academic_results'])} academic results"
            )

            return state

        except Exception as e:
            logger.error(f"Error in search node: {traceback.format_exc()}")
            state["error"] = str(e)
            return state

    async def _summary_node(self, state: MultiAgentState) -> MultiAgentState:
        """Summary node."""
        try:
            logger.info("Executing summary node")

            # Check if we skipped processing due to security threats
            if not state.get("is_safe", True):
                logger.info("Creating security-aware summary for unsafe query")
                state["summary"] = (
                    f"Security analysis detected a {state.get('threat_level', 'unknown')} level threat in the query. The query has been sanitized and processed safely. No research results were generated due to security concerns."
                )

                # Create a minimal research result
                state["research_result"] = ResearchResult(
                    search_query=state["research_query"],
                    summary=state["summary"],
                    papers=[],
                    sources=["security_analysis"],
                    total_found=0,
                    search_time=0.0,
                    error=state.get("error"),
                )

                state["current_step"] = "completed"
                return state

            # Normal summary processing for safe queries
            summary_state: SummaryState = {
                "papers": state["papers"],
                "web_results": state["web_results"],
                "academic_results": state["academic_results"],
                "query": state["research_query"],
                "summary": "",
                "research_result": None,
                "error": None,
            }

            result_state = await self.agents["summary"].process_state(summary_state)

            state["summary"] = result_state["summary"]
            state["research_result"] = result_state["research_result"]
            state["current_step"] = "completed"
            state["error"] = result_state.get("error")

            logger.info("Summary completed")

            return state

        except Exception as e:
            logger.error(f"Error in summary node: {traceback.format_exc()}")
            state["error"] = str(e)
            return state

    @staticmethod
    def add_papers_to_vector_store(papers: list) -> None:
        """Add papers to the vector store for future similarity search."""
        try:
            logger.info(f"Adding {len(papers)} papers to vector store")
            vector_store.add_papers(papers)
            vector_store.save_index()
            logger.info("Successfully added papers to vector store")

        except Exception:
            logger.error(f"Error adding papers to vector store: {traceback.format_exc()}")


# Global multi-agent orchestrator instance
multi_agent_orchestrator = MultiAgentOrchestrator()
