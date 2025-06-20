from dotenv import load_dotenv
load_dotenv()
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List, Optional, Dict, Any

from graph import create_graph
from models import AgentState

app = FastAPI(
    title="LangGraph data driven interruption template",
    version="1.0",
    description="A LangGraph template to show how to check for required fields and interrupt the graph if fields are missing.",
)

graph = create_graph()

# Add a simple test endpoint
@app.post("/test")
async def test_graph(input_data: Dict[str, Any] = None):
    """Test endpoint to verify the graph is working."""
    try:
        if input_data is None:
            input_data = {"input": "I want to analyze a document"}
            
        print(f"Received input data: {input_data}")
        
        # Run the graph with the input data
        result = graph.invoke(input_data)
        
        print(f"Graph result: {result}")
        
        return result
    except Exception as e:
        print(f"Error in test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Configure the graph for LangServe
add_routes(
    app,
    graph.with_config({"recursion_limit": 50}),  # Add config to prevent infinite loops
    path="/graph",
    playground_type="default",
    enable_feedback_endpoint=True,
    per_req_config_modifier=lambda config: {
        **config,
        "recursion_limit": 50,
        "configurable": {"session_id": "test-session"}
    }
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the LangGraph data-driven interruption template.",
        "endpoints": {
            "test": "/test (POST - test the graph directly)",
            "playground": "/graph/playground",
            "invoke": "/graph/invoke",
            "stream": "/graph/stream"
        },
        "example_input": {
            "input": "I want to analyze a document"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI server on http://127.0.0.1:{port}")
    print(f"Test the graph directly: curl -X POST http://127.0.0.1:{port}/test")
    print(f"Or visit http://127.0.0.1:{port}/graph/playground to use the playground")
    uvicorn.run(app, host="127.0.0.1", port=port)