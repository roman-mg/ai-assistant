# AI Research Assistant

An intelligent chatbot assistant that can research the newest AI/ML papers and provide links + summaries. Built with LangGraph, FastAPI, FAISS, and OpenAI.

## Features

- 🔍 **Intelligent Research**: Automatically searches ArXiv for relevant AI/ML papers
- 📊 **Paper Analysis**: Uses OpenAI to analyze and summarize papers
- 🧠 **Vector Search**: FAISS-powered similarity search for finding related papers
- 💬 **Chat Interface**: RESTful API and WebSocket support for real-time chat
- 🔄 **Streaming Responses**: Real-time streaming of research results
- 📈 **Conversation Memory**: Maintains conversation history and research context

## Tech Stack

- **LangGraph**: Agent workflow orchestration
- **FastAPI**: High-performance web framework
- **FAISS**: Vector similarity search
- **OpenAI**: LLM and embeddings
- **Python 3.12**: Modern Python features

## Quick Start

### 1. Install Dependencies

```bash
# Install using pip (recommended)
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

### 2. Set Up Environment

The application uses default configuration values, but you can override them by setting environment variables:

```bash
# Set your OpenAI API key (required)
export MODEL_OPENAI_API_KEY="your_openai_api_key_here"

# Optional: Override other settings
export MODEL_OPENAI_MODEL="gpt-4o"
export RESEARCHER_MAX_PAPERS_PER_QUERY=10
export VECTOR_STORE_FAISS_INDEX_PATH="./data/faiss_index"
```

### 3. Run the Application

```bash
# Quick start with the startup script
python src.main

# Or run the main module
python src/main.py
```

The API will be available at `http://localhost:8000`

### 4. Try the API

Visit `http://localhost:8000/docs` for interactive API documentation.

You can also test the API using the provided example script:

```bash
# Run the example usage script
python example_usage.py
```

## API Endpoints

### Chat Endpoints

- `POST /chat` - Send a research query and get results
- `POST /chat/stream` - Streaming version of chat endpoint
- `WebSocket /ws` - Real-time WebSocket chat

### Utility Endpoints

- `GET /health` - Health check
- `GET /papers/count` - Number of papers in vector store
- `POST /papers/search` - Search similar papers

## Usage Examples

### Basic Chat Request

```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "What are the latest advances in transformer architectures?",
    "stream": False
})

result = response.json()
print(result["message"])
```

### Streaming Chat

```python
import requests

response = requests.post("http://localhost:8000/chat/stream", json={
    "message": "Recent developments in reinforcement learning",
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    ws.send(JSON.stringify({
        message: "What are the newest papers on computer vision?",
        conversation_id: "unique-conversation-id"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

## Project Structure

```
src/
├── agents/           # LangGraph agents
│   └── research_agent.py
├── api/              # FastAPI application
│   └── main.py
├── config/           # Configuration management
│   └── settings.py
├── di/               # Dependency injection
│   └── fabric.py
├── models/           # Pydantic schemas
│   └── schemas.py
├── services/         # Shared services
│   └── llm_service.py
├── tools/            # Research tools
│   ├── arxiv_tool.py
│   ├── web_search_tool.py
│   └── paper_analyzer.py
├── vectorstore/      # FAISS vector store
│   └── faiss_store.py
└── main.py           # Application entry point
```

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Loguru** for elegant logging

```bash
# Check codestyle
poe hard-check

# Format code
poe format
```

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
poe test
```

## Configuration

All configuration is managed through environment variables with sensible defaults. Key settings:

- `MODEL_OPENAI_API_KEY`: Your OpenAI API key (required)
- `MODEL_OPENAI_MODEL`: OpenAI model to use (default: gpt-4o)
- `MODEL_OPENAI_EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `RESEARCHER_MAX_PAPERS_PER_QUERY`: Maximum papers to return per query (default: 10)
- `VECTOR_STORE_FAISS_INDEX_PATH`: Path to store FAISS index (default: ./data/faiss_index)
- `WEB_SEARCH_ENABLED`: Enable web search functionality (default: true)
- `APP_DEBUG`: Enable debug mode (default: false)

## Architecture

The application uses a multi-agent architecture with shared services:

1. **LLM Service**: Shared OpenAI LLM and embeddings service for consistency
2. **Research Agent**: LangGraph-based agent that orchestrates the research workflow
3. **ArXiv Tool**: Searches academic papers from ArXiv
4. **Web Search Tool**: Additional web search capabilities
5. **Paper Analyzer**: Uses shared LLM to analyze and summarize papers
6. **Vector Store**: FAISS-based similarity search using shared embeddings
7. **FastAPI Backend**: RESTful API and WebSocket endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.