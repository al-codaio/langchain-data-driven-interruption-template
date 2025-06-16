from typing import List, Optional, Dict, Any, Annotated, Sequence, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode

class AgentState(BaseModel):
    """State for the agent."""
    messages: List[BaseMessage] = Field(default_factory=list)
    user_email: Optional[str] = None
    document_id: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    next_node_after_validation: Optional[str] = None
    current_plan: str = ""
    analysis_result: Optional[str] = None

# Node requirements dictionary
NODE_REQUIREMENTS = {
    "start_analysis": ["user_email", "document_id"],
    "perform_step_2": ["analysis_result"],
}