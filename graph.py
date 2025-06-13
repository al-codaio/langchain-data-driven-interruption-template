import os
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import AgentState, NODE_REQUIREMENTS

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.5)

def validate_state_and_interrupt(state: AgentState) -> Command[str] | AgentState:
    """
    Checks if required fields for the *next intended node* are present.
    Interrupts if missing and generates a user message.
    """
    print(f"\n--- Running validate_state_and_interrupt ---")
    
    target_node = state.next_node_after_validation
    
    if not target_node:
        print("Error: next_node_after_validation not set. This node should always be preceded by a node setting this.")
        return Command(goto="orchestrator", update={"messages": state.messages + [AIMessage(content="I encountered an error. Let me start over.")]})

    required_fields_for_next = NODE_REQUIREMENTS.get(target_node, [])
    
    missing = []
    for field in required_fields_for_next:
        if not hasattr(state, field) or getattr(state, field) is None or (isinstance(getattr(state, field), str) and not getattr(state, field)):
            missing.append(field)
            
    state.missing_fields = missing

    if missing:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Based on the missing fields, generate a polite message asking the user to provide the information. List the missing fields clearly and suggest how to provide them, e.g., 'Email: your@email.com, Document ID: 123'. Focus on one missing field at a time if there are many, or prioritize based on importance."""),
            ("user", "The following fields are missing: {missing_fields}. Please ask the user to provide them.")
        ])
        llm_chain = prompt_template | llm
        
        missing_str = ", ".join(m.replace("_", " ") for m in missing)
        llm_response = llm_chain.invoke({"missing_fields": missing_str}).content
        
        user_prompt_message = f"I need some more information to proceed.\n{llm_response}"
        print(f"Generated user prompt: {user_prompt_message}")

        if state.messages:
            state.messages.append(AIMessage(content=user_prompt_message))
        else:
            state.messages = [AIMessage(content=user_prompt_message)]

        print(f"Interrupting for missing fields: {missing}")
        return Command(
            interrupt=user_prompt_message,
            update={
                "messages": state.messages,
                "missing_fields": missing,
                "next_node_after_validation": target_node
            }
        )
    else:
        print("All required fields present. Proceeding to target node.")
        state.missing_fields = []
        return Command(goto=target_node, update={"missing_fields": []})


def start_analysis(state: AgentState) -> AgentState:
    print(f"\n--- Running start_analysis node ---")
    print(f"Performing analysis for email: {state.user_email}, document: {state.document_id}")
    analysis_result = f"Analysis complete for {state.user_email} on doc {state.document_id}. Results: [DUMMY DATA]"
    state.messages.append(AIMessage(content=analysis_result))
    state.current_plan = "Analysis completed. Now proceeding to Step 2."
    state.analysis_result = analysis_result # Set a new field that perform_step_2 might need
    return state


def perform_step_2(state: AgentState) -> AgentState:
    print(f"\n--- Running perform_step_2 node ---")
    print(f"Using analysis result: {state.analysis_result}")
    step_2_result = f"Step 2 completed based on {state.analysis_result[:30]}..."
    state.messages.append(AIMessage(content=step_2_result))
    state.current_plan = "Step 2 completed."
    return state


def process_user_input(state: AgentState) -> AgentState:
    """
    This node processes the user's response after an interrupt.
    It expects the user to provide the missing fields.
    """
    print(f"\n--- Running process_user_input node ---")
    
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        user_response = last_message.content
        print(f"User response received: '{user_response}'")

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that extracts information from user messages.
             Extract the following fields from the user's message: {fields_to_extract}.
             Respond with a JSON object. If a field is not found, omit it.
             Example: {{"email": "user@example.com", "document_id": "doc123"}}
             """),
            ("user", "User message: {user_message}")
        ])
        extraction_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.0)
        
        fields_to_extract_str = ", ".join(state.missing_fields)
        
        try:
            import json
            llm_extraction_result = extraction_llm.invoke({
                "fields_to_extract": fields_to_extract_str,
                "user_message": user_response
            }).content
            
            provided_info = json.loads(llm_extraction_result.strip().strip('`').strip('json'))
        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM extraction/JSON parse error: {e}. Raw LLM output: {llm_extraction_result}")
            provided_info = {}

        print(f"Extracted info: {provided_info}")

        updated_any_field = False
        if "user_email" in provided_info and provided_info["user_email"] and not state.user_email:
            state.user_email = provided_info["user_email"]
            updated_any_field = True
            print(f"Updated user_email: {state.user_email}")
        if "document_id" in provided_info and provided_info["document_id"] and not state.document_id:
            state.document_id = provided_info["document_id"]
            updated_any_field = True
            print(f"Updated document_id: {state.document_id}")
            
        if not updated_any_field and state.missing_fields:
            state.messages.append(AIMessage(content="I'm sorry, I couldn't find the requested information in your message. Please try again. For example, 'Email: your@email.com, Document ID: 123'."))
        elif updated_any_field:
            state.messages.append(AIMessage(content="Thank you! Let me check those details."))
            state.missing_fields = []
            
    return state

def orchestrator_node(state: AgentState) -> Command[str] | AgentState:
    """
    Decides the next logical step in the graph based on the current state and plan.
    Sets `state.next_node_after_validation` before routing to `validate_state`.
    """
    print(f"\n--- Running orchestrator_node ---")
    print(f"Current state: {state.dict()}")
    
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, HumanMessage):
        print(f"Processing human message: {last_message.content}")
        
        # For the first message, always try to start analysis
        if len(state.messages) == 1:
            state.next_node_after_validation = "start_analysis"
            state.current_plan = "Starting analysis process."
            return Command(goto="validate_state")
            
        # For subsequent messages, use LLM to decide
        if not state.missing_fields:
            llm_decision_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an intelligent assistant. Based on the current conversation history and existing plan, determine the *next logical step* for the user's request.
                Your response should be a single string from the following options:
                - "start_analysis": If the user's intent is to analyze a document or similar.
                - "perform_step_2": If analysis seems complete and the next logical step is a follow-up action.
                - "end_conversation": If the task is complete or the user expresses a desire to end.
                - "clarify_intent": If the user's intent is unclear or requires more information beyond specific fields.

                Current plan: {current_plan}
                Conversation history: {history}
                """),
                ("user", "User's current message: {latest_message}")
            ])
            
            llm_chain = llm_decision_prompt | llm
            
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.messages[:-1]])
            
            decision = llm_chain.invoke({
                "current_plan": state.current_plan,
                "history": history_str,
                "latest_message": last_message.content
            }).content.strip().lower()

            print(f"Orchestrator decided: {decision}")
            
            if "start_analysis" in decision:
                state.next_node_after_validation = "start_analysis"
                state.current_plan = "Starting analysis process."
                return Command(goto="validate_state")
            elif "perform_step_2" in decision:
                state.next_node_after_validation = "perform_step_2"
                state.current_plan = "Proceeding with second step."
                return Command(goto="validate_state")
            elif "end_conversation" in decision:
                state.messages.append(AIMessage(content="It was a pleasure assisting you. Goodbye!"))
                return Command(goto="output_handler")
            elif "clarify_intent" in decision:
                state.messages.append(AIMessage(content="I'm not sure how to proceed. Could you please clarify what you'd like me to do?"))
                state.current_plan = "Awaiting clarification."
                return state
            else:
                state.messages.append(AIMessage(content="I'm sorry, I couldn't determine the next step. Could you please rephrase your request?"))
                state.current_plan = "Awaiting rephrased request."
                return state

    if state.next_node_after_validation:
        return Command(goto="validate_state")
    else:
        if not state.messages or not isinstance(state.messages[-1], HumanMessage):
            state.messages.append(AIMessage(content="Hello! How can I assist you today?"))
        return state

def create_graph():
    """Create and return the graph for LangServe."""
    workflow = StateGraph(AgentState)
    
    # Add input handling node
    def handle_input(state: Dict[str, Any]) -> AgentState:
        """Transform input into AgentState."""
        print(f"\n--- Running handle_input ---")
        print(f"Received input state: {state}")
        
        # Handle both string input and dict with 'input' key
        input_text = state.get('input', state) if isinstance(state, dict) else state
        
        # Create initial state with the input message
        initial_state = AgentState(
            messages=[HumanMessage(content=str(input_text))],
            current_plan="Starting new conversation"
        )
        
        print(f"Created initial state: {initial_state.dict()}")
        return initial_state
    
    # Add output handling node
    def handle_output(state: AgentState) -> Dict[str, Any]:
        """Transform AgentState into output format."""
        print(f"\n--- Running handle_output ---")
        print(f"Output state: {state.dict()}")
        
        if not state.messages:
            return {"output": "No response generated", "state": state.dict()}
        
        last_message = state.messages[-1]
        if isinstance(last_message, (HumanMessage, AIMessage)):
            return {
                "output": last_message.content,
                "state": state.dict(),
                "type": "message"
            }
        return {
            "output": str(last_message),
            "state": state.dict(),
            "type": "text"
        }
    
    # Add nodes
    workflow.add_node("input_handler", handle_input)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("validate_state", validate_state_and_interrupt)
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("start_analysis", start_analysis)
    workflow.add_node("perform_step_2", perform_step_2)
    workflow.add_node("output_handler", handle_output)
    
    # Add edges
    workflow.add_edge("input_handler", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: x.goto if isinstance(x, Command) else "output_handler",
        {
            "validate_state": "validate_state",
            "output_handler": "output_handler"
        }
    )
    workflow.add_conditional_edges(
        "validate_state",
        lambda x: "process_user_input" if isinstance(x, Command) and x.interrupt else x.goto if isinstance(x, Command) else "orchestrator",
        {
            "process_user_input": "process_user_input",
            "start_analysis": "start_analysis",
            "perform_step_2": "perform_step_2",
            "orchestrator": "orchestrator"
        }
    )
    workflow.add_edge("process_user_input", "validate_state")
    workflow.add_edge("start_analysis", "output_handler")
    workflow.add_edge("perform_step_2", "output_handler")
    
    # Set entry point
    workflow.set_entry_point("input_handler")
    
    return workflow.compile()

app = create_graph()