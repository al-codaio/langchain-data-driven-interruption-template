from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, root_validator
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    messages: List[dict] = Field(default_factory=list)
    user_email: Optional[str] = None
    document_id: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    next_node_after_validation: Optional[str] = None
    current_plan: str = "" 
    analysis_result: Optional[str] = None

NODE_REQUIREMENTS = {
    "start_analysis": ["user_email", "document_id"],
    "perform_step_2": ["analysis_result"],
}

class FlexibleInput(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

    @root_validator
    def at_least_one(cls, values):
        if not values.get("message") and not values.get("messages"):
            raise ValueError("Either 'message' or 'messages' must be provided.")
        return values

def flexible_input_to_agent_state(input_data: FlexibleInput) -> AgentState:
    if input_data.messages:
        return AgentState(messages=input_data.messages)
    if input_data.message:
        return AgentState(messages=[{"type": "human", "content": input_data.message}])
    return AgentState()