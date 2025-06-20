from dotenv import load_dotenv
load_dotenv()
import os
import uuid
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Mount static files for serving HTML and other assets
app.mount("/static", StaticFiles(directory="."), name="static")

graph = create_graph()

# Configure the graph for LangServe
def per_req_config_modifier(config, request):
    """Generate configurable keys for each request."""
    # Use a consistent session ID to maintain conversation state
    # In production, you'd get this from the request headers or user session
    session_id = "test-session-123"  # Fixed session ID for testing
    
    return {
        **config,
        "recursion_limit": 50,
        "configurable": {
            "thread_id": f"thread-{session_id}",
            "checkpoint_ns": "langgraph-app",
            "checkpoint_id": f"checkpoint-{session_id}"
        }
    }

add_routes(
    app,
    graph.with_config({"recursion_limit": 50}),  # Add config to prevent infinite loops
    path="/graph",
    playground_type="default",
    enable_feedback_endpoint=True,
    per_req_config_modifier=per_req_config_modifier
)

# Add a session-based endpoint for better production use
@app.post("/analyze")
async def analyze_with_session(input_data: Dict[str, Any]):
    """Production endpoint that properly handles sessions and configurable keys."""
    try:
        # Generate a stable session ID based on user input or request
        session_id = str(uuid.uuid4())
        
        config = {
            "configurable": {
                "thread_id": f"user-session-{session_id}",
                "checkpoint_ns": "production-namespace",
                "checkpoint_id": f"session-{session_id}"
            }
        }
        
        result = graph.invoke(input_data, config=config)
        return {
            "result": result,
            "session_id": session_id,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to the LangGraph data-driven interruption template.",
        "endpoints": {
            "chat": "/chat (GET - Chat interface)",
            "analyze": "/analyze (POST - production endpoint with session management)",
            "playground": "/graph/playground",
            "invoke": "/graph/invoke",
            "stream": "/graph/stream"
        },
        "example_input": {
            "input": "I want to analyze a document"
        }
    }

# Serve the chat interface
@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Serve the chat interface HTML."""
    try:
        with open("chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat interface not found")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI server on http://127.0.0.1:{port}")
    print(f"Visit http://127.0.0.1:{port}/graph/playground to use the playground")
    uvicorn.run(app, host="127.0.0.1", port=port)