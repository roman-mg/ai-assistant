"""Paper analyzer tool using OpenAI for summarization and analysis."""

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from src.models.schemas import Paper
from src.services.llm_service import llm_service


class PaperAnalysisInput(BaseModel):
    """Input for paper analysis tool."""

    papers: list[Paper] = Field(..., description="list of papers to analyze")
    analysis_type: str = Field(
        default="summary",
        description="Type of analysis: summary, key_points, methodology, or comparison",
    )
    max_summary_length: int = Field(
        default=500,
        description="Maximum length of summary in characters",
        ge=100,
        le=2000,
    )


class PaperAnalyzerTool(BaseTool):
    """Tool for analyzing and summarizing research papers using OpenAI."""

    name: str = "paper_analyzer"
    description: str = (
        "Analyze and summarize research papers using AI. This tool can provide "
        "summaries, extract key points, analyze methodologies, or compare papers. "
        "Use this tool to make papers more accessible and understandable."
    )
    args_schema: type[BaseModel] = PaperAnalysisInput

    def __init__(self):
        super().__init__()

    def _run(
        self,
        papers: list[Paper],
        analysis_type: str = "summary",
        max_summary_length: int = 500,
    ) -> list[Paper]:
        """Analyze papers and return enhanced versions with summaries."""
        if not papers:
            return []

        try:
            logger.info(f"Analyzing {len(papers)} papers with analysis type: {analysis_type}")
            
            analyzed_papers = []
            for paper in papers:
                try:
                    analyzed_paper = self._analyze_single_paper(
                        paper, analysis_type, max_summary_length
                    )
                    analyzed_papers.append(analyzed_paper)
                except Exception as e:
                    logger.warning(f"Error analyzing paper {paper.title}: {e}")
                    # Return original paper if analysis fails
                    analyzed_papers.append(paper)

            logger.info(f"Successfully analyzed {len(analyzed_papers)} papers")
            return analyzed_papers

        except Exception as e:
            logger.error(f"Error in paper analysis: {e}")
            return papers  # Return original papers if analysis fails

    def _analyze_single_paper(
        self,
        paper: Paper,
        analysis_type: str,
        max_summary_length: int,
    ) -> Paper:
        """Analyze a single paper."""
        try:
            # Create analysis prompt based on type
            prompt = self._create_analysis_prompt(paper, analysis_type, max_summary_length)
            
            # Get analysis from LLM
            summary = llm_service.invoke_chat(prompt)

            # Create enhanced paper with summary
            enhanced_paper = Paper(
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                summary=summary,
                url=paper.url,
                published_date=paper.published_date,
                categories=paper.categories,
                similarity_score=paper.similarity_score,
            )

            return enhanced_paper

        except Exception as e:
            logger.error(f"Error analyzing paper {paper.title}: {e}")
            return paper

    def _create_analysis_prompt(
        self,
        paper: Paper,
        analysis_type: str,
        max_summary_length: int,
    ) -> str:
        """Create analysis prompt based on type."""
        base_prompt = f"""
        Please analyze the following research paper and provide a {analysis_type}:

        Title: {paper.title}
        Authors: {', '.join(paper.authors)}
        Abstract: {paper.abstract}

        Please provide a concise {analysis_type} in no more than {max_summary_length} characters.
        Focus on the most important aspects and make it accessible to a general audience.
        """

        if analysis_type == "summary":
            return base_prompt + """
            Summarize the key contributions, methodology, and findings of this paper.
            """
        elif analysis_type == "key_points":
            return base_prompt + """
            Extract the main key points, contributions, and important findings.
            """
        elif analysis_type == "methodology":
            return base_prompt + """
            Focus on the methodology, approach, and technical details used in this research.
            """
        elif analysis_type == "comparison":
            return base_prompt + """
            Highlight what makes this paper unique compared to other work in the field.
            """
        else:
            return base_prompt + """
            Provide a general analysis of this research paper.
            """

    async def _arun(
        self,
        papers: list[Paper],
        analysis_type: str = "summary",
        max_summary_length: int = 500,
    ) -> list[Paper]:
        """Async version of paper analysis."""
        return self._run(papers, analysis_type, max_summary_length)


class PaperComparisonInput(BaseModel):
    """Input for paper comparison tool."""

    papers: list[Paper] = Field(..., description="list of papers to compare", min_items=2)
    comparison_aspects: list[str] = Field(
        default=["methodology", "contributions", "results"],
        description="Aspects to compare",
    )


class PaperComparisonTool(BaseTool):
    """Tool for comparing multiple research papers."""

    name: str = "paper_comparison"
    description: str = (
        "Compare multiple research papers across different aspects like methodology, "
        "contributions, and results. Useful for understanding relationships between papers "
        "and identifying trends in research."
    )
    args_schema: type[BaseModel] = PaperComparisonInput

    def __init__(self):
        super().__init__()

    def _run(
        self,
        papers: list[Paper],
        comparison_aspects: list[str] = None,
    ) -> str:
        """Compare multiple papers and return comparison analysis."""
        if len(papers) < 2:
            return "At least 2 papers are required for comparison."

        if comparison_aspects is None:
            comparison_aspects = ["methodology", "contributions", "results"]

        try:
            logger.info(f"Comparing {len(papers)} papers across {comparison_aspects}")
            
            # Create comparison prompt
            prompt = self._create_comparison_prompt(papers, comparison_aspects)
            
            # Get comparison from LLM
            comparison = llm_service.invoke_chat(prompt)

            logger.info("Successfully generated paper comparison")
            return comparison

        except Exception as e:
            logger.error(f"Error comparing papers: {e}")
            return f"Error comparing papers: {str(e)}"

    def _create_comparison_prompt(
        self,
        papers: list[Paper],
        comparison_aspects: list[str],
    ) -> str:
        """Create comparison prompt."""
        papers_info = []
        for i, paper in enumerate(papers, 1):
            papers_info.append(f"""
            Paper {i}:
            Title: {paper.title}
            Authors: {', '.join(paper.authors)}
            Abstract: {paper.abstract}
            """)

        prompt = f"""
        Please compare the following research papers across these aspects: {', '.join(comparison_aspects)}

        {''.join(papers_info)}

        Provide a detailed comparison that:
        1. Identifies similarities and differences
        2. Highlights unique contributions of each paper
        3. Discusses how the papers relate to each other
        4. Identifies trends or patterns across the papers
        5. Suggests which papers might be most relevant for different use cases

        Keep the comparison concise but comprehensive, focusing on the most important insights.
        """

        return prompt

    async def _arun(
        self,
        papers: list[Paper],
        comparison_aspects: list[str] = None,
    ) -> str:
        """Async version of paper comparison."""
        return self._run(papers, comparison_aspects)
