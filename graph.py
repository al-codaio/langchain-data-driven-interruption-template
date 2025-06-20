import os
import json
import traceback
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import AgentState, NODE_REQUIREMENTS

print("=== GRAPH.PY MODULE LOADED ===")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.5)

def test_function():
    print("=== TEST FUNCTION CALLED ===")
    return "test"

def test_validate_function():
    print("=== TESTING VALIDATE FUNCTION ===")
    test_state = AgentState(
        messages=[HumanMessage(content="test")],
        next_node_after_validation="start_analysis"
    )
    result = validate_state_and_interrupt(test_state)
    print(f"=== VALIDATE FUNCTION RESULT: {result} ===")
    return result

def test_node(state: AgentState) -> AgentState:
    print("=== TEST NODE CALLED ===")
    state.messages.append(AIMessage(content="Test node was called!"))
    return state

def validate_state_and_interrupt(state: AgentState) -> Command[str] | AgentState:
    """
    Checks if required fields for the *next intended node* are present.
    Interrupts if missing and generates a user message.
    """
    print(f"\n=== VALIDATE_STATE_AND_INTERRUPT CALLED ===")
    print(f"FUNCTION ENTRY POINT REACHED")
    print(f"FUNCTION DEFINITION LOADED SUCCESSFULLY")
    print(f"\n--- Running validate_state_and_interrupt ---")
    print(f"DEBUG: validate_state_and_interrupt called with state: {state.dict()}")
    
    # Super simple test - just return the state with a test message
    print(f"TEST: About to append test message")
    state.messages.append(AIMessage(content="SUPER TEST: validate_state_and_interrupt was called!"))
    print(f"TEST: Function executed successfully, returning state")
    return state

def simple_validate_function(state: AgentState) -> AgentState:
    """Simple test function to replace validate_state_and_interrupt"""
    print(f"\n=== SIMPLE_VALIDATE_FUNCTION CALLED ===")
    print(f"Simple function was called with state: {state}")
    state.messages.append(AIMessage(content="SIMPLE TEST: simple_validate_function was called!"))
    print(f"Simple function executed successfully")
    return state

def ultra_simple_function(state: AgentState) -> AgentState:
    """Ultra simple test function"""
    print(f"\n=== ULTRA SIMPLE FUNCTION CALLED ===")
    print(f"ULTRA SIMPLE: Function was called!")
    state.messages.append(AIMessage(content="ULTRA SIMPLE: Function was called!"))
    return state

print("=== VALIDATE_STATE_AND_INTERRUPT FUNCTION DEFINED ===")

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
    print(f"[DEBUG] process_user_input entered with missing_fields: {state.missing_fields}")
    
    last_message = state.messages[-1]
    print(f"[DEBUG] last_message type: {type(last_message)}, content: {getattr(last_message, 'content', None)}")
    is_human = False
    if isinstance(last_message, HumanMessage):
        is_human = True
    elif isinstance(last_message, BaseMessage) and getattr(last_message, 'type', None) == 'human':
        is_human = True

    if not state.missing_fields and state.next_node_after_validation:
        from models import NODE_REQUIREMENTS
        state.missing_fields = NODE_REQUIREMENTS.get(state.next_node_after_validation, [])

    if is_human:
        user_response = last_message.content
        print(f"User response received: '{user_response}'")

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that extracts information from user messages.\nExtract the following fields from the user's message: {fields_to_extract}.\nRespond ONLY with a valid JSON object using the field names as keys. If a field is not found, omit it.\nDo not include any explanation or extra text.\n"""),
            ("user", "User message: {user_message}")
        ])
        extraction_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.0)
        
        fields_to_extract_str = ", ".join(state.missing_fields)
        print(f"[DEBUG] fields_to_extract_str before extraction: {fields_to_extract_str}")
        if fields_to_extract_str:
            prompt_value = None
            llm_response_obj = None
            llm_extraction_result = None
            updated_any_field = False
            try:
                print(f"[DEBUG] Extraction prompt input: fields_to_extract={fields_to_extract_str}, user_message={user_response}")
                prompt_value = extraction_prompt.format_prompt(fields_to_extract=fields_to_extract_str, user_message=user_response)
                print(f"[DEBUG] prompt_value assigned")
                llm_response_obj = extraction_llm.invoke(prompt_value)
                print(f"[DEBUG] llm_response_obj assigned")
                print(f"[DEBUG] Full LLM response object: {llm_response_obj!r}")
                llm_extraction_result = getattr(llm_response_obj, "content", None)
                print(f"[DEBUG] Raw LLM extraction result: {llm_extraction_result!r}")
                provided_info = json.loads(llm_extraction_result.strip().strip('`').strip('json'))
                print(f"Extracted info: {provided_info}")
                user_email = provided_info.get("user_email")
                if user_email and not state.user_email:
                    state.user_email = user_email
                    updated_any_field = True
                if "document_id" in provided_info and provided_info["document_id"] and not state.document_id:
                    print(f"[DEBUG] Setting state.document_id = {provided_info['document_id']}")
                    state.document_id = provided_info["document_id"]
                    updated_any_field = True
                
                if not updated_any_field and state.missing_fields:
                    state.messages.append(AIMessage(content="I'm sorry, I couldn't find the requested information in your message. Please try again. For example, 'Email: your@email.com, Document ID: 123'."))
                    return state
                elif updated_any_field:
                    state.messages.append(AIMessage(content="Thank you! Let me check those details."))
                    state.missing_fields = []
                    # Route to orchestrator to continue the flow, and persist updated fields
                    return Command(
                        goto="orchestrator",
                        update={
                            "user_email": state.user_email,
                            "document_id": state.document_id,
                            "missing_fields": [],
                            "messages": state.messages,
                        }
                    )
            except Exception as e:
                print(f"LLM extraction error: {e!r}")
                traceback.print_exc()
                if prompt_value is not None:
                    print(f"[DEBUG] prompt_value: {prompt_value!r}")
                if llm_response_obj is not None:
                    print(f"[DEBUG] llm_response_obj: {llm_response_obj!r}")
                if llm_extraction_result is not None:
                    print(f"[DEBUG] llm_extraction_result: {llm_extraction_result!r}")
                provided_info = {}

    return state

def orchestrator_node(state: AgentState) -> Command[str] | AgentState:
    print(f"[DEBUG] orchestrator_node: current_plan = {state.current_plan}")
    
    # Check if we just completed start_analysis and should proceed to perform_step_2
    if state.current_plan.startswith("Analysis completed") and state.analysis_result:
        print(f"[DEBUG] Analysis completed, routing directly to perform_step_2")
        return Command(goto="perform_step_2")
    
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

    # Check if this is a response to an interruption (AI message asking for missing fields)
    if is_human and len(state.messages) >= 2:
        prev_message = state.messages[-2]
        prev_is_ai = getattr(prev_message, "type", None) == "ai"
        prev_content = getattr(prev_message, "content", "")
        # Check for interruption patterns in the previous AI message
        if prev_is_ai and any(phrase in prev_content for phrase in ["I need some more information", "Could you please provide", "missing fields"]):
            from models import NODE_REQUIREMENTS
            next_node = state.next_node_after_validation or "start_analysis"
            required_fields = NODE_REQUIREMENTS.get(next_node, [])
            missing = [field for field in required_fields if getattr(state, field, None) in (None, "")]
            print(f"[DEBUG] Orchestrator detected interruption response, routing to process_user_input with missing_fields: {missing}")
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
            print(f"[DEBUG] First message detected, routing to validate_state")
            # Update the state instead of returning a command
            state.next_node_after_validation = "start_analysis"
            state.current_plan = "Starting analysis process."
            print(f"[DEBUG] Orchestrator updated state and will continue to next node")
            return state
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
                    goto="validate_state_test",
                    update={
                        "next_node_after_validation": "start_analysis",
                        "current_plan": "Starting analysis process."
                    }
                )
            elif "perform_step_2" in decision:
                return Command(
                    goto="validate_state_test",
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
        return Command(goto="validate_state_test")
    else:
        if not state.messages or not (is_human):
            state.messages.append(AIMessage(content="Hello! How can I assist you today?"))
        return Command(goto="output_handler")

def output_handler(state: AgentState) -> AgentState:
    """
    Handles the final output of the graph.
    """
    print(f"\n--- Running output_handler ---")
    
    # Check if this is an interruption case
    if state.missing_fields:
        print(f"Interruption detected with missing fields: {state.missing_fields}")
        # Return the last AI message (the interruption message)
        if state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, AIMessage):
                return AgentState(
                    messages=state.messages,
                    user_email=state.user_email,
                    document_id=state.document_id,
                    missing_fields=state.missing_fields,
                    next_node_after_validation=state.next_node_after_validation,
                    current_plan=state.current_plan,
                    analysis_result=state.analysis_result
                )
    
    # Normal output handling
    if state.messages:
        last_message = state.messages[-1]
        print(f"Last message: {last_message}")
        print(f"Last message type: {type(last_message)}")
        print(f"Last message content: {last_message.content}")
        
        if isinstance(last_message, AIMessage):
            return state
        else:
            # If the last message is from human, we need to generate a response
            fallback_output = AIMessage(content=last_message.content)
            print(f"Fallback output: {fallback_output}")
            state.messages.append(fallback_output)
            return state
    else:
        # No messages in state, this shouldn't happen
        fallback_message = AIMessage(content="I'm not sure how to respond to that.")
        return AgentState(
            messages=[fallback_message],
            user_email=state.user_email,
            document_id=state.document_id,
            missing_fields=state.missing_fields,
            next_node_after_validation=state.next_node_after_validation,
            current_plan=state.current_plan,
            analysis_result=state.analysis_result
        )

def create_graph():
    """Create and return the graph for LangServe."""
    # Create StateGraph with custom input schema
    workflow = StateGraph(AgentState)
    
    # Configure the graph to accept string input
    workflow.set_entry_point("orchestrator")
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("test_node", test_node)
    print("=== ADDING VALIDATE_STATE NODE ===")
    workflow.add_node("validate_state_test", ultra_simple_function)
    print("=== VALIDATE_STATE NODE ADDED ===")
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("start_analysis", start_analysis)
    workflow.add_node("perform_step_2", perform_step_2)
    workflow.add_node("output_handler", output_handler)

    # Edges
    def orchestrator_router(state):
        print(f"[DEBUG] Orchestrator router called with state: {state}")
        print(f"[DEBUG] next_node_after_validation: {getattr(state, 'next_node_after_validation', None)}")
        if hasattr(state, 'next_node_after_validation') and state.next_node_after_validation:
            print(f"[DEBUG] Routing to validate_state_test")
            return "validate_state_test"
        else:
            print(f"[DEBUG] Routing to output_handler")
            return "output_handler"
    
    # Temporarily use direct edge instead of conditional edge
    print("=== ADDING DIRECT EDGE FROM ORCHESTRATOR TO VALIDATE_STATE_TEST ===")
    workflow.add_edge("orchestrator", "validate_state_test")
    print("=== DIRECT EDGE ADDED ===")
    
    def validate_state_router(result):
        print(f"[DEBUG] Validate_state router called with result: {result}, type: {type(result)}")
        print(f"[DEBUG] Validate_state router - result is Command: {isinstance(result, Command)}")
        print(f"[DEBUG] Validate_state router - ABOUT TO ROUTE")
        if not isinstance(result, Command):
            print(f"[DEBUG] Routing to output_handler (interruption case)")
            return "output_handler"
        else:
            print(f"[DEBUG] Routing to {result.goto}")
            return result.goto
    
    workflow.add_conditional_edges(
        "validate_state_test",
        validate_state_router,
        {
            "output_handler": "output_handler",
            "start_analysis": "start_analysis",
            "perform_step_2": "perform_step_2",
            "orchestrator": "orchestrator"
        }
    )
    
    workflow.add_edge("process_user_input", "orchestrator")
    workflow.add_edge("test_node", "output_handler")
    workflow.add_edge("start_analysis", "output_handler")
    workflow.add_edge("perform_step_2", "output_handler")

    # Set up checkpointer when compiling the graph
    checkpointer = MemorySaver()  # In-memory; swap for persistent in production
    
    # Compile the graph
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["validate_state_test"],
        interrupt_after=["validate_state_test"]
    )
    
    return app

app = create_graph()

# Test the validate function
if __name__ == "__main__":
    test_validate_function()