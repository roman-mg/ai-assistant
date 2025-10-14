"""FAISS vector store for paper similarity search and retrieval."""

import os
import pickle

import faiss
import numpy as np
from loguru import logger

from ..config.settings import settings
from ..models.schemas import Paper
from ..services.llm_service import llm_service


class FAISSPaperStore:
    """FAISS-based vector store for research papers."""

    def __init__(self):
        self.index: faiss.Index | None = None
        self.papers: list[Paper] = []
        self.dimension = settings.vector_store.vector_dimension
        self.index_path = settings.vector_store.faiss_index_path
        self.papers_path = os.path.join(os.path.dirname(self.index_path), "papers.pkl")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Load existing index if available
        self._load_index()

    def _load_index(self) -> None:
        """Load existing FAISS index and papers from disk."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.papers_path):
                logger.info("Loading existing FAISS index and papers")

                # Load FAISS index
                self.index = faiss.read_index(self.index_path)

                # Load papers
                with open(self.papers_path, "rb") as f:
                    self.papers = pickle.load(f)

                logger.info(f"Loaded {len(self.papers)} papers from existing index")
            else:
                logger.info("No existing index found, creating new one")
                self._create_new_index()

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.papers = []
            logger.info("Created new FAISS index")

        except Exception as e:
            logger.error(f"Error creating new index: {e}")
            raise

    def add_papers(self, papers: list[Paper]) -> None:
        """Add papers to the vector store."""
        if not papers:
            return

        try:
            logger.info(f"Adding {len(papers)} papers to vector store")

            # Generate embeddings for papers
            texts = [self._paper_to_text(paper) for paper in papers]
            embeddings = llm_service.embed_documents(texts)

            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype("float32")

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)

            # Add to FAISS index
            self.index.add(embeddings_array)

            # Add papers to list
            self.papers.extend(papers)

            logger.info(f"Successfully added {len(papers)} papers to vector store")

        except Exception as e:
            logger.error(f"Error adding papers to vector store: {e}")
            raise

    def search_similar_papers(
        self,
        query: str,
        k: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[tuple[Paper, float]]:
        """Search for similar papers using vector similarity."""
        try:
            if not self.papers or self.index is None:
                logger.warning("No papers in vector store")
                return []

            logger.info(f"Searching for papers similar to: {query}")

            # Generate embedding for query
            query_embedding = llm_service.embed_query(query)
            query_vector = np.array([query_embedding]).astype("float32")

            # Normalize query vector
            faiss.normalize_L2(query_vector)

            # Search in FAISS index
            scores, indices = self.index.search(query_vector, min(k, len(self.papers)))

            # Filter results by similarity threshold
            threshold = similarity_threshold or settings.vector_store.similarity_threshold
            results = []

            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx >= 0 and score >= threshold:  # Valid index and above threshold
                    paper = self.papers[idx]
                    # Add similarity score to paper
                    paper_with_score = Paper(
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        summary=paper.summary,
                        url=paper.url,
                        published_date=paper.published_date,
                        categories=paper.categories,
                        similarity_score=float(score),
                    )
                    results.append((paper_with_score, float(score)))

            logger.info(f"Found {len(results)} similar papers")
            return results

        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []

    def get_paper_by_title(self, title: str) -> Paper | None:
        """Get a paper by its title."""
        for paper in self.papers:
            if paper.title.lower() == title.lower():
                return paper
        return None

    def get_papers_by_category(self, category: str) -> list[Paper]:
        """Get all papers in a specific category."""
        return [paper for paper in self.papers if category.lower() in [cat.lower() for cat in paper.categories]]

    def get_recent_papers(self, days: int = 30) -> list[Paper]:
        """Get papers published within the last N days."""
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        return [paper for paper in self.papers if paper.published_date and paper.published_date >= cutoff_date]

    def get_paper_count(self) -> int:
        """Get total number of papers in the store."""
        return len(self.papers)

    def save_index(self) -> None:
        """Save FAISS index and papers to disk."""
        try:
            if self.index is not None:
                logger.info("Saving FAISS index and papers to disk")

                # Save FAISS index
                faiss.write_index(self.index, self.index_path)

                # Save papers
                with open(self.papers_path, "wb") as f:
                    pickle.dump(self.papers, f)

                logger.info("Successfully saved index and papers")

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise

    @staticmethod
    def _paper_to_text(paper: Paper) -> str:
        """Convert paper to text for embedding."""
        text_parts = [
            paper.title,
            " ".join(paper.authors),
            paper.abstract,
            " ".join(paper.categories),
        ]
        return " ".join(text_parts)

    def clear_index(self) -> None:
        """Clear all papers from the index."""
        try:
            logger.info("Clearing vector store index")
            self._create_new_index()

            # Remove saved files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.papers_path):
                os.remove(self.papers_path)

            logger.info("Successfully cleared vector store")

        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise

    def rebuild_index(self) -> None:
        """Rebuild the entire index from scratch."""
        try:
            logger.info("Rebuilding vector store index")

            if not self.papers:
                logger.warning("No papers to rebuild index with")
                return

            # Create new index
            self._create_new_index()

            # Re-add all papers
            papers_to_rebuild = self.papers.copy()
            self.papers.clear()
            self.add_papers(papers_to_rebuild)

            logger.info("Successfully rebuilt vector store index")

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise


# Global vector store instance
vector_store = FAISSPaperStore()
