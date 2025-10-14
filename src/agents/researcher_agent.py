"""LangGraph research agent for orchestrating research workflows."""

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from ..config.settings import settings
from ..models.schemas import Paper, ResearchResult
from ..services.llm_service import llm_service
from ..tools.arxiv_tool import ArxivTool, RecentPapersTool
from ..tools.paper_analyzer_tool import PaperAnalyzerTool, PaperComparisonTool
from ..tools.web_search_tool import AcademicSearchTool, WebSearchTool
from ..vectorstore.faiss_store import vector_store


class AgentState(TypedDict):
    """State for the research agent."""

    messages: list[dict[str, Any]]
    papers: list[Paper]
    web_results: list[dict]
    research_query: str
    research_results: ResearchResult | None
    current_step: str
    error: str | None


class ResearcherAgent:
    """LangGraph-based research agent for AI/ML paper research."""

    def __init__(self):
        # Initialize tools
        self.tools = [
            ArxivTool(),
            RecentPapersTool(),
            PaperAnalyzerTool(),
            PaperComparisonTool(),
            WebSearchTool(),
            AcademicSearchTool(),
        ]

        # Create tool node
        self.tool_node = ToolNode(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    async def research(
        self,
        query: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> ResearchResult:
        """Perform research on a given query."""
        logger.info(f"Starting research for query: {query}")
        start_time = time.time()
        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": conversation_history or [],
                "papers": [],
                "web_results": [],
                "research_query": query,
                "research_results": None,
                "current_step": "analyze_query",
                "error": None,
            }

            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Calculate search time
            search_time = time.time() - start_time

            # Create research result
            if final_state.get("research_results"):
                result = final_state["research_results"]
                result.search_time = search_time
            else:
                result = ResearchResult(
                    papers=final_state.get("papers", []),
                    total_found=len(final_state.get("papers", [])),
                    search_query=query,
                    search_time=search_time,
                    sources=["arxiv", "vector_store", "web_search"],
                )

            logger.info(f"Research completed in {search_time:.2f}s, found {len(result.papers)} papers")
            return result

        except Exception as e:
            logger.error(f"Error in research: {e}")
            return ResearchResult(
                papers=[],
                total_found=0,
                search_query=query,
                search_time=time.time() - start_time,
                sources=[],
                error=str(e),
            )

    @staticmethod
    def add_papers_to_vector_store(papers: list[Paper]) -> None:
        """Add papers to the vector store for future similarity search."""
        try:
            logger.info(f"Adding {len(papers)} papers to vector store")
            vector_store.add_papers(papers)
            vector_store.save_index()
            logger.info("Successfully added papers to vector store")

        except Exception as e:
            logger.error(f"Error adding papers to vector store: {e}")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("search_papers", self._search_papers)
        workflow.add_node("analyze_papers", self._analyze_papers)
        workflow.add_node("search_vector_store", self._search_vector_store)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.set_entry_point("analyze_query")

        workflow.add_edge("analyze_query", "search_papers")
        workflow.add_edge("search_papers", "analyze_papers")
        workflow.add_edge("analyze_papers", "search_vector_store")
        workflow.add_edge("search_vector_store", "web_search")
        workflow.add_edge("web_search", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    @staticmethod
    async def _analyze_query( state: AgentState) -> AgentState:
        """Analyze the research query to determine search strategy."""
        try:
            query = state["research_query"]
            logger.info(f"Analyzing query: {query}")

            # Use LLM to analyze query and determine search strategy
            analysis_prompt = f"""
            Analyze this research query and determine the best search strategy:
            
            Query: {query}
            
            Consider:
            1. Is this asking for recent papers or general research?
            2. What specific AI/ML topics are mentioned?
            3. Should we search for recent papers or general papers?
            4. What ArXiv categories might be relevant?
            
            Respond with a JSON object containing:
            - search_type: "recent" or "general"
            - categories: list of relevant ArXiv categories
            - keywords: list of key terms to search for
            - max_papers: suggested number of papers to find
            """

            response = await llm_service.ainvoke_chat(analysis_prompt)

            # Parse response (simplified - in production, use proper JSON parsing)
            state["current_step"] = "search_papers"

            logger.info("Query analysis completed")
            return state

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            state["error"] = str(e)
            return state

    @staticmethod
    async def _search_papers(state: AgentState) -> AgentState:
        """Search for papers using ArXiv."""
        try:
            query = state["research_query"]
            logger.info(f"Searching papers for: {query}")

            # Use ArXiv tool to search for papers
            arxiv_tool = ArxivTool()
            papers = await arxiv_tool._arun(
                query=query,
                max_results=settings.researcher.max_papers_per_query,
            )

            state["papers"] = papers
            state["current_step"] = "analyze_papers"

            logger.info(f"Found {len(papers)} papers from ArXiv")
            return state

        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            state["error"] = str(e)
            return state

    @staticmethod
    async def _analyze_papers(state: AgentState) -> AgentState:
        """Analyze and summarize the found papers."""
        try:
            papers = state["papers"]
            if not papers:
                logger.warning("No papers to analyze")
                state["current_step"] = "search_vector_store"
                return state

            logger.info(f"Analyzing {len(papers)} papers")

            # Use paper analyzer tool
            analyzer_tool = PaperAnalyzerTool()
            analyzed_papers = await analyzer_tool._arun(
                papers=papers,
                analysis_type="summary",
                max_summary_length=settings.researcher.max_paper_summary_length,
            )

            state["papers"] = analyzed_papers
            state["current_step"] = "search_vector_store"

            logger.info(f"Successfully analyzed {len(analyzed_papers)} papers")
            return state

        except Exception as e:
            logger.error(f"Error analyzing papers: {e}")
            state["error"] = str(e)
            return state

    @staticmethod
    async def _search_vector_store(state: AgentState) -> AgentState:
        """Search for similar papers in the vector store."""
        try:
            query = state["research_query"]
            logger.info(f"Searching vector store for: {query}")

            # Search vector store for similar papers
            similar_papers = vector_store.search_similar_papers(
                query=query,
                k=5,  # Get top 5 similar papers
                similarity_threshold=settings.vector_store.similarity_threshold,
            )

            # Add similar papers to existing papers (avoid duplicates)
            existing_titles = {paper.title.lower() for paper in state["papers"]}
            for paper, score in similar_papers:
                if paper.title.lower() not in existing_titles:
                    state["papers"].append(paper)

            state["current_step"] = "web_search"

            logger.info(f"Found {len(similar_papers)} similar papers in vector store")
            return state

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            state["error"] = str(e)
            return state

    @staticmethod
    async def _web_search(state: AgentState) -> AgentState:
        """Perform additional web search for context."""
        try:
            if not settings.web_search.enabled:
                logger.info("Web search disabled, skipping")
                state["current_step"] = "generate_response"
                return state

            query = state["research_query"]
            logger.info(f"Performing web search for: {query}")

            # Use web search tool
            web_tool = WebSearchTool()
            web_results = await web_tool._arun(
                query=query,
                max_results=settings.web_search.max_results,
            )

            # Store web results in state for response generation
            state["web_results"] = web_results
            state["current_step"] = "generate_response"

            logger.info(f"Found {len(web_results)} web search results")
            return state

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            state["error"] = str(e)
            return state

    @staticmethod
    async def _generate_response(state: AgentState) -> AgentState:
        """Generate final response with research results."""
        try:
            query = state["research_query"]
            papers = state["papers"]
            web_results = state.get("web_results", [])

            logger.info("Generating final response")

            # Create research result
            research_result = ResearchResult(
                papers=papers,
                total_found=len(papers),
                search_query=query,
                search_time=0,  # Will be set by caller
                sources=["arxiv", "vector_store", "web_search"] if web_results else ["arxiv", "vector_store"],
            )

            state["research_results"] = research_result
            state["current_step"] = "completed"

            logger.info("Response generation completed")
            return state

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state["error"] = str(e)
            return state


# Global research agent instance
researcher_agent = ResearcherAgent()
