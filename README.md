# AI Research Assistant

An intelligent multi-agent research system that provides comprehensive AI/ML research capabilities with advanced security, query analysis, and search functionality. Built with LangGraph, FastAPI, MCP (Model Context Protocol), and OpenAI.

## 🏗️ Architecture Overview

The system uses a **4-Agent Multi-Agent Architecture** with specialized agents working in sequence:

```
Security → Query Analysis → Search → Summary → Response
```

### 🔒 **Security Agent**
- **Purpose**: First-line defense against prompt injection and malicious inputs
- **Features**: Multi-level threat detection, input sanitization, LLM-enhanced analysis
- **Output**: Sanitized input and safety status

### 🔍 **Query Analysis Agent** 
- **Purpose**: Analyzes and summarizes user queries in structured JSON format
- **Features**: Query optimization, academic focus detection, structured output
- **Output**: JSON-formatted query analysis with main topic, focus area, and key terms

### 🔎 **Search Agent**
- **Purpose**: Comprehensive research across multiple sources
- **Features**: ArXiv search, vector store similarity search, MCP-based web search
- **Output**: Papers, web results, and academic content

### 📝 **Summary Agent**
- **Purpose**: Creates comprehensive research summaries and final results
- **Features**: Multi-source synthesis, research result generation, metadata enrichment
- **Output**: Structured research results with summaries and citations

## ✨ Key Features

- 🛡️ **Advanced Security**: Multi-layer prompt injection protection with threat classification
- 🧠 **Intelligent Query Analysis**: Structured JSON query analysis and optimization
- 🔍 **Multi-Source Search**: ArXiv papers, vector similarity, and MCP-based web search
- 📊 **Comprehensive Summarization**: AI-powered synthesis of research findings
- 🔄 **Streaming Responses**: Real-time streaming of research results
- 💬 **Chat Interface**: RESTful API and WebSocket support
- 📈 **Conversation Memory**: Maintains research context across conversations
- 🎯 **Academic Focus**: Specialized academic search capabilities

## 🛠️ Tech Stack

- **LangGraph**: Multi-agent workflow orchestration
- **FastAPI**: High-performance web framework
- **MCP**: Model Context Protocol for web search
- **FAISS**: Vector similarity search
- **OpenAI**: LLM and embeddings
- **Python 3.12**: Modern Python features

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install using pip (recommended)
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

### 2. Set Up Environment

```bash
# Set your OpenAI API key (required)
export MODEL_OPENAI_API_KEY="your_openai_api_key_here"

# Optional: Override other settings
export MODEL_OPENAI_MODEL="gpt-4o"
export RESEARCHER_MAX_PAPERS_PER_QUERY=10
export VECTOR_STORE_FAISS_INDEX_PATH="./data/faiss_index"
export WEB_SEARCH_ENABLED=true
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

## 📡 API Endpoints

### Chat Endpoints

- `POST /chat` - Send a research query and get results
- `POST /chat/stream` - Streaming version of chat endpoint
- `WebSocket /ws` - Real-time WebSocket chat

### Utility Endpoints

- `GET /health` - Health check
- `GET /papers/count` - Number of papers in vector store
- `POST /papers/search` - Search similar papers

## 💡 Usage Examples

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

## 🏛️ Project Structure

```
src/
├── agents/                    # Multi-agent system
│   ├── multi_agent_orchestrator.py  # Main orchestrator
│   ├── security_agent.py           # Security & threat detection
│   ├── query_analysis_agent.py     # Query analysis & JSON formatting
│   ├── search_agent.py             # Multi-source search coordination
│   └── summary_agent.py           # Result synthesis & summarization
├── api/                      # FastAPI application
│   └── main.py
├── config/                   # Configuration management
│   └── settings.py
├── di/                      # Dependency injection
│   └── fabric.py
├── models/                  # Pydantic schemas
│   └── schemas.py
├── services/               # Shared services
│   └── llm_service.py
├── tools/                  # Research tools
│   ├── arxiv_tool.py       # ArXiv paper search
│   ├── web_search_tool.py  # MCP-based web search
│   └── paper_analyzer_tool.py
├── vectorstore/            # FAISS vector store
│   └── faiss_store.py
└── main.py                # Application entry point
```

## 🔧 Multi-Agent Workflow

### 1. Security Analysis
```python
# Input validation and threat detection
security_state = {
    "original_input": user_query,
    "sanitized_input": cleaned_query,
    "is_safe": True/False,
    "threat_level": "low/medium/high/critical",
    "detected_threats": ["pattern1", "pattern2"]
}
```

### 2. Query Analysis
```python
# Structured query analysis
query_analysis = {
    "main_topic": "machine learning",
    "focus_area": "neural networks",
    "key_terms": ["transformer", "attention", "deep learning"],
    "query_summary": "User is asking about transformer architectures"
}
```

### 3. Comprehensive Search
```python
# Multi-source research
search_results = {
    "papers": [arxiv_papers],
    "web_results": [wikipedia_articles],
    "vector_results": [similar_papers]
}
```

### 4. Intelligent Summarization
```python
# AI-powered synthesis
research_result = {
    "query": original_query,
    "summary": comprehensive_summary,
    "papers": found_papers,
    "sources": ["arxiv", "wikipedia", "vector_store"],
    "total_found": count,
    "search_time": duration
}
```

## 🛡️ Security Features

### Threat Detection Levels
- **Low**: Minor suspicious patterns
- **Medium**: Potential manipulation attempts  
- **High**: Clear injection attempts
- **Critical**: Dangerous system commands

### Protection Mechanisms
- **Pattern Matching**: Regex-based threat detection
- **Keyword Filtering**: Dangerous word identification
- **LLM Analysis**: AI-powered threat assessment
- **Input Sanitization**: Automatic cleaning of malicious content
- **Workflow Blocking**: Unsafe queries skip processing

## 🔍 Search Capabilities

### ArXiv Integration
- Real-time paper search
- Category filtering
- Recent papers discovery
- Metadata extraction

### Vector Store Search
- FAISS-powered similarity search
- Semantic paper matching
- Duplicate detection
- Relevance scoring

### MCP Web Search
- Wikipedia integration
- Academic content focus
- Structured data extraction
- No API key requirements

## ⚙️ Configuration

All configuration is managed through environment variables with sensible defaults:

### Required Settings
- `MODEL_OPENAI_API_KEY`: Your OpenAI API key

### Optional Settings
- `MODEL_OPENAI_MODEL`: OpenAI model (default: gpt-4o)
- `MODEL_OPENAI_EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `RESEARCHER_MAX_PAPERS_PER_QUERY`: Max papers per query (default: 10)
- `VECTOR_STORE_FAISS_INDEX_PATH`: FAISS index path (default: ./data/faiss_index)
- `WEB_SEARCH_ENABLED`: Enable web search (default: true)
- `APP_DEBUG`: Debug mode (default: false)

## 🧪 Development

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

## 🔄 Agent Communication

The multi-agent system uses **LangGraph** for orchestration:

```python
# Workflow definition
builder = StateGraph(MultiAgentState)
builder.add_node("security", security_node)
builder.add_node("query_analysis", query_analysis_node) 
builder.add_node("search", search_node)
builder.add_node("summary", summary_node)

# Conditional routing
builder.add_conditional_edges(
    "security",
    should_continue_after_security,
    {"continue": "query_analysis", "skip_to_summary": "summary"}
)
```

## 🎯 Key Benefits

1. **Security First**: Multi-layer protection against malicious inputs
2. **Intelligent Analysis**: Structured query understanding and optimization
3. **Comprehensive Research**: Multiple search sources and methodologies
4. **AI-Powered Synthesis**: Intelligent summarization and result generation
5. **Scalable Architecture**: Modular agent design for easy extension
6. **Real-time Processing**: Streaming responses and WebSocket support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using LangGraph, FastAPI, and OpenAI**