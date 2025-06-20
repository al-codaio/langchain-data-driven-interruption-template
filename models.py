from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage

class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    user_email: Optional[str] = None
    document_id: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    next_node_after_validation: Optional[str] = None
    current_plan: str = "" 
    analysis_result: Optional[str] = None

class StringInput(BaseModel):
    input: str

NODE_REQUIREMENTS = {
    "start_analysis": ["user_email", "document_id"],
    "perform_step_2": ["analysis_result"],
}