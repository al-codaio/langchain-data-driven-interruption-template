import os
import json
import traceback
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
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
            ("system", """You are a helpful assistant. Based on the missing fields, 
            generate a polite message asking the user to provide the information. 
            Keep the message short and concise. List the missing fields clearly and 
            suggest how to provide them, e.g., 'Email: your@email.com, Document ID: 123'. 
            Focus on one missing field at a time if there are many, or prioritize based on importance.
            """),
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
        state.missing_fields = missing
        return interrupt(user_prompt_message)
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
    Process user input to collect missing required fields.
    Accepts either a direct value (e.g., "myemail@example.com") or a JSON object.
    """
    if not state.missing_fields:
        return state

    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state

    user_input = last_message.content.strip()
    
    # Try to parse as JSON first
    try:
        if user_input.startswith('{'):
            data = json.loads(user_input)
            # Update state with any provided fields
            for field in state.missing_fields:
                if field in data:
                    setattr(state, field, data[field])
        else:
            # If not JSON, assume it's a direct value for the first missing field
            setattr(state, state.missing_fields[0], user_input)
    except json.JSONDecodeError:
        # If JSON parsing fails, treat as direct value
        setattr(state, state.missing_fields[0], user_input)

    # Check if we still have missing fields
    still_missing = [field for field in state.missing_fields 
                    if not getattr(state, field)]
    
    if still_missing:
        # Ask for the next missing field
        state.messages.append(AIMessage(
            content=f"Please provide your {still_missing[0].replace('_', ' ')}"
        ))
        state.missing_fields = still_missing
        return state
    
    # All fields collected, proceed
    state.messages.append(AIMessage(content="Thank you! Let me process that."))
    state.missing_fields = []
    return Command(
        goto="orchestrator",
        update={
            "user_email": state.user_email,
            "document_id": state.document_id,
            "missing_fields": [],
            "messages": state.messages,
        }
    )

def orchestrator_node(state: AgentState) -> Command[str] | AgentState:
    print(f"[DEBUG] orchestrator_node: current_plan = {state.current_plan}")
    # QUICK FIX: If analysis is completed, route to output_handler
    if state.current_plan.startswith("Analysis completed"):
        return Command(goto="output_handler")
    """
    Decides the next logical step in the graph based on the current state and plan.
    Sets `state.next_node_after_validation` before routing to `validate_state`.
    """
    print(f"\n--- Running orchestrator_node ---")
    print(f"Current state: {state.dict()}")
    print(f"orchestrator_node: state.messages = {state.messages}")
    print(f"orchestrator_node: len(state.messages) = {len(state.messages)}")
    for i, m in enumerate(state.messages):
        print(f"  message[{i}]: type={getattr(m, 'type', None)}, content={getattr(m, 'content', None)}")

    # Generalized: After any AI interruption for missing fields, next human message routes to process_user_input
    is_human = False
    if len(state.messages) >= 1:
        last_message = state.messages[-1]
        is_human = getattr(last_message, "type", None) == "human"

    if is_human and len(state.messages) >= 2:
        prev_message = state.messages[-2]
        prev_is_ai = getattr(prev_message, "type", None) == "ai"
        prev_content = getattr(prev_message, "content", "")
        if prev_is_ai and "I need some more information to proceed." in prev_content:
            from models import NODE_REQUIREMENTS
            next_node = state.next_node_after_validation or "start_analysis"
            required_fields = NODE_REQUIREMENTS.get(next_node, [])
            missing = [field for field in required_fields if getattr(state, field, None) in (None, "")]
            print(f"[DEBUG] Orchestrator set missing_fields: {missing} and next_node_after_validation: {next_node}")
            print(f"[DEBUG] Orchestrator about to return to process_user_input with missing_fields: {missing}")
            return Command(
                goto="process_user_input",
                update={
                    "missing_fields": missing,
                    "next_node_after_validation": next_node
                }
            )

    if is_human:
        print(f"Processing human message: {getattr(last_message, 'content', None)}")
        # Only set next_node_after_validation if it is not already set (i.e., is None) and on the very first user message
        if len(state.messages) == 1 and not state.next_node_after_validation:
            return Command(
                goto="validate_state",
                update={
                    "next_node_after_validation": "start_analysis",
                    "current_plan": "Starting analysis process."
                }
            )
        # For subsequent messages, use LLM to decide
        if not state.missing_fields:
            llm_decision_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an intelligent assistant. Based on the current conversation history and existing plan, determine the *next logical step* for the user's request.\n                Your response should be a single string from the following options:\n                - \"start_analysis\": If the user's intent is to analyze a document or similar.\n                - \"perform_step_2\": If analysis seems complete and the next logical step is a follow-up action.\n                - \"end_conversation\": If the task is complete or the user expresses a desire to end.\n                - \"clarify_intent\": If the user's intent is unclear or requires more information beyond specific fields.\n\n                Current plan: {current_plan}\n                Conversation history: {history}\n                """),
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
                return Command(
                    goto="validate_state",
                    update={
                        "next_node_after_validation": "start_analysis",
                        "current_plan": "Starting analysis process."
                    }
                )
            elif "perform_step_2" in decision:
                return Command(
                    goto="validate_state",
                    update={
                        "next_node_after_validation": "perform_step_2",
                        "current_plan": "Proceeding with second step."
                    }
                )
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
        if not state.messages or not (is_human):
            state.messages.append(AIMessage(content="Hello! How can I assist you today?"))
        return state

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

def create_graph():
    workflow = StateGraph(AgentState)
    # Add nodes (no input_handler)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("validate_state", validate_state_and_interrupt)
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("start_analysis", start_analysis)
    workflow.add_node("perform_step_2", perform_step_2)
    workflow.add_node("output_handler", handle_output)

    # Edges (no edge from input_handler)
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
    workflow.add_edge("process_user_input", "orchestrator")
    workflow.add_edge("start_analysis", "output_handler")
    workflow.add_edge("perform_step_2", "output_handler")

    # Set entry point to orchestrator
    workflow.set_entry_point("orchestrator")

    return workflow.compile()

app = create_graph()