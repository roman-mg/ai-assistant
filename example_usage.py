"""Example script demonstrating how to use the AI Research Assistant API."""

import asyncio
import json

import requests
import websockets
from loguru import logger


def test_chat_endpoint():
    """Test the basic chat endpoint."""
    logger.info("Testing chat endpoint...")
    
    url = "http://localhost:8000/chat"
    data = {
        "message": "What are the latest advances in transformer architectures?",
        "stream": False,
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.success(f"Response: {result['message']}")
        logger.info(f"Papers found: {len(result['research_results']['papers'])}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


def test_streaming_endpoint():
    """Test the streaming chat endpoint."""
    logger.info("Testing streaming endpoint...")
    
    url = "http://localhost:8000/chat/stream"
    data = {
        "message": "Recent developments in reinforcement learning",
        "stream": True,
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        logger.info("Streaming response:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    chunk_data = json.loads(line_str[6:])
                    logger.info(f"Chunk: {chunk_data['content']}")
                    
    except Exception as e:
        logger.error(f"Error: {e}")


async def test_websocket():
    """Test the WebSocket endpoint."""
    logger.info("Testing WebSocket endpoint...")
    
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            message = {
                "message": "What are the newest papers on computer vision?",
                "conversation_id": "test-conversation",
            }
            
            await websocket.send(json.dumps(message))
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                logger.info(f"WebSocket response: {data}")
                
                if data.get('type') == 'response':
                    break
                    
    except Exception as e:
        logger.error(f"Error: {e}")


def test_paper_search():
    """Test the paper search endpoint."""
    logger.info("Testing paper search endpoint...")
    
    url = "http://localhost:8000/papers/search"
    data = {
        "query": "machine learning",
        "limit": 5,
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        logger.success(f"Found {result['total_found']} papers")
        for paper in result['papers'][:3]:
            logger.info(f"- {paper['title']}")
            
    except Exception as e:
        logger.error(f"Error: {e}")


def test_health_check():
    """Test the health check endpoint."""
    logger.info("Testing health check...")
    
    try:
        response = requests.get("http://localhost:8000/health")
        response.raise_for_status()
        
        result = response.json()
        logger.success(f"Status: {result['status']}")
        logger.info(f"Version: {result['version']}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


def main():
    """Run all tests."""
    logger.info("AI Research Assistant API Tests")
    logger.info("=" * 40)
    
    # Test health check first
    test_health_check()
    
    # Test other endpoints
    test_chat_endpoint()
    test_streaming_endpoint()
    test_paper_search()
    
    # Test WebSocket (async)
    logger.info("Testing WebSocket (async)...")
    asyncio.run(test_websocket())
    
    logger.success("All tests completed!")


if __name__ == "__main__":
    main()
