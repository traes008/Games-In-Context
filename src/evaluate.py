"""
Evaluation module for assessing LLM decision-making in game-theoretic scenarios.

This module implements a LangGraph-based agent architecture that evaluates how different
LLMs make decisions when faced with strategic games. Agents are prompted to reason through
scenarios and make tool-based decisions, with results captured for analysis.
"""

import copy
import json
import os
import csv
import logging
import operator
from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END

from . import config
from . import model

# Configure module logger
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INPUT_DIR: str = getattr(config, "SCENARIOS_PATH", "scenarios")
RESULTS_DIR: str = getattr(config, "RESULTS_PATH", "results")
EVALUATIONS_DIR: str = getattr(config, "EVALUATIONS_PATH", "results/evaluations")


def _get_eval_model_config(model_config: Optional[Dict[str, str]] = None) -> tuple[str, str]:
    """
    Extract evaluation model configuration.
    
    Args:
        model_config: Optional dict with 'name' and 'provider' keys.
                      If None, falls back to the first entry in EVAL_MODELS.
    
    Returns:
        Tuple of (provider, model_name).
    """
    if model_config:
        provider = model_config.get("provider", "ollama")
        name = model_config.get("name", "mistral-nemo:12b")
        logger.info(f"Using evaluation model: {provider}/{name}")
        return provider, name
    models = getattr(config, "EVAL_MODELS", None)
    if isinstance(models, list) and models:
        first = models[0]
        if isinstance(first, dict):
            provider = first.get("provider", "ollama")
            name = first.get("name", "mistral-nemo:12b")
            logger.info(f"Using evaluation model: {provider}/{name}")
            return provider, name
    logger.warning("No evaluation models configured, using default")
    return "ollama", "mistral-nemo:12b"


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename on Windows.
    
    Args:
        name: Original filename string.
        
    Returns:
        Sanitized filename with only safe characters.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .")
    cleaned = "".join(ch if ch in allowed else "_" for ch in name)
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "model"


# These module-level defaults are set at import time and can be overridden
# per-run by evaluate(model_config=...).
MODEL_PROVIDER, MODEL_NAME = _get_eval_model_config()
OUTPUT_FILE = os.path.join(
    EVALUATIONS_DIR,
    f"results_{_sanitize_filename(MODEL_NAME)}.csv"
)

# Ensure directories exist
os.makedirs(EVALUATIONS_DIR, exist_ok=True)

# Debugging: read TEST_FIRST_N from src/config.py (set e.g. TEST_FIRST_N = 3)
TEST_FIRST_N = None
try:
    TEST_FIRST_N = getattr(config, "TEST_FIRST_N", None)
    if TEST_FIRST_N is not None:
        TEST_FIRST_N = int(TEST_FIRST_N)
        logger.info(f"Testing first {TEST_FIRST_N} scenarios only (debug mode)")
except Exception as e:
    logger.warning(f"Could not read TEST_FIRST_N: {e}")
    TEST_FIRST_N = None


# --- 1. LOAD PERSONAS ---
def _load_experiment_personas() -> Dict[str, str]:
    """
    Load persona definitions from configuration.
    
    Returns:
        Dictionary mapping persona names to persona prompts.
    """
    personas = getattr(config, "PERSONAS", None)
    if isinstance(personas, dict) and personas:
        logger.info(f"Loaded {len(personas)} personas from config")
        return personas
    
    logger.warning("No personas configured, using default")
    return {
        "Default_Assistant": "You are a helpful, harmless, and honest AI assistant."
    }


PERSONAS = _load_experiment_personas()


# --- 2. LANGGRAPH STATE & NODES ---

class GameState(TypedDict):
    scenario_config: dict
    persona_prompt: str
    messages: Annotated[List[BaseMessage], operator.add] 
    final_decision: Optional[dict]
    retry_count: int

def agent_node(state: GameState) -> Dict[str, Any]:
    """
    Agent node that invokes the LLM to respond to a game scenario.
    
    The agent is given a persona prompt, game scenario, and available tools,
    and is instructed to make a decision via tool calling.
    
    Args:
        state: Current game state.
        
    Returns:
        Updated state with the LLM's response message.
    """
    scenario = state["scenario_config"]
    persona = state["persona_prompt"]
    
    try:
        # 1. Initialize LLM
        llm = model.LLMFactory.get(MODEL_PROVIDER, MODEL_NAME, temperature=config.TEMPERATURE_EVALUATION)

        # 2. Generate/Bind Tools
        # Inject a 'reasoning' parameter into every tool so the model
        # is forced to articulate its reasoning as a structured argument.
        tools = []
        for t in scenario.get("tools", []):
            params = t.get("parameters", {"type": "object", "properties": {}})
            
            # Deep-copy to avoid mutating the original scenario data
            params = copy.deepcopy(params)
            
            # Ensure properties exists
            if "properties" not in params:
                params["properties"] = {}

            # Add reasoning parameter
            params["properties"]["reasoning"] = {
                "type": "string",
                "description": "Your step-by-step reasoning for this decision."
            }
            # Keep reasoning optional to avoid tool-call failures
            if "required" not in params:
                params["required"] = []

            tool_def = {
                "name": t["name"],
                "description": t["description"],
                "parameters": params
            }
            tools.append(tool_def)
        
        llm_with_tools = llm.bind_tools(tools)
        
        # 3. Construct Messages
        # Google's API requires at least one user message in contents.
        # Split: persona → SystemMessage, scenario/task → HumanMessage.
        system_msg = SystemMessage(content=persona)
        user_msg = HumanMessage(
            content=f"CONTEXT:\n{scenario.get('description')}\n\n"
                    f"TASK:\nThink through your decision step-by-step, then submit "
                    f"your final decision by calling the provided tool. You MUST "
                    f"include your reasoning in the 'reasoning' parameter of the tool call."
        )

        print(f"Human message to LLM:\n{user_msg.content}\n")
                    
        # 4. Invoke
        messages = [system_msg, user_msg] + state.get("messages", [])
        response = llm_with_tools.call(messages)
        
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in agent_node: {e}", exc_info=True)
        response = HumanMessage(content=f"Error invoking model: {e}")
        return {"messages": [response]}

def tool_node(state: GameState) -> Dict[str, Any]:
    """
    Process the LLM's tool call and extract the final decision.
    
    Extracts reasoning from the message content and tool call arguments,
    then constructs a decision object with action, args, value, and reasoning.
    
    Args:
        state: Current game state.
        
    Returns:
        Updated state with final_decision set if tool was called, else None.
    """
    last_message = state["messages"][-1]
    
    try:
        # 1. Extract reasoning text from the LLM's response
        response_text = getattr(last_message, 'content', '')
        content_reasoning = ""
        
        # Handle standard string content
        if isinstance(response_text, str):
            content_reasoning = response_text.strip()
            
        # Handle list content (LangChain's format for mixed text + tools)
        elif isinstance(response_text, list):
            text_blocks = [
                block.get("text", "") for block in response_text 
                if isinstance(block, dict) and "text" in block
            ]
            content_reasoning = "\n".join(text_blocks).strip()
            
        # 2. Process the tool call and final decision
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            
            action = tool_call["name"]
            args = dict(tool_call.get("args", {}))
            
            # Extract reasoning from tool args first (most reliable),
            # then fall back to message content
            tool_reasoning = args.pop("reasoning", "").strip()
            reasoning = tool_reasoning or content_reasoning or "No reasoning provided."
            
            value = args.get("number") if "number" in args else action

            decision = {
                "action": action,
                "args": args,
                "value": value,
                "reasoning": reasoning
            }
            logger.debug(f"Tool call processed: {action}")
            return {"final_decision": decision}
        
        logger.debug("No tool call found in LLM response")
        return {"final_decision": None}
        
    except Exception as e:
        logger.error(f"Error in tool_node: {e}", exc_info=True)
        return {"final_decision": None}

def retry_node(state: GameState) -> Dict[str, Any]:
    """
    Handle retry logic when LLM fails to call a tool.
    
    Prompts the LLM again to submit a tool call.
    
    Args:
        state: Current game state.
        
    Returns:
        Updated state with retry message and incremented retry count.
    """
    max_retries = getattr(config, "MAX_RETRY_COUNT", 2)
    logger.debug(f"Retry attempt {state['retry_count'] + 1}/{max_retries}")
    scenario = state.get("scenario_config", {})
    prompt = (
        "You did not call a tool. You MUST use a tool to submit your decision. "
    )
    return {
        "messages": [HumanMessage(content=prompt)],
        "retry_count": state["retry_count"] + 1
    }


def should_continue(state: GameState) -> str:
    """
    Conditional router for continuing or ending the agent loop.
    
    Routes to:
    - 'end_success': If a valid decision was made.
    - 'retry': If retry count not exceeded and no decision yet.
    - 'end_fail': If max retries exceeded without valid decision.
    
    Args:
        state: Current game state.
        
    Returns:
        Next node identifier.
    """
    decision = state.get("final_decision")
    max_retries = getattr(config, "MAX_RETRY_COUNT", 2)
    
    if decision:
        return "end_success"
    if state["retry_count"] >= max_retries:
        logger.debug("Max retries exceeded, ending with failure")
        return "end_fail"
    return "retry"


# --- 3. GRAPH COMPILATION ---

def build_graph():
    """
    Build the LangGraph workflow for game-theoretic agent evaluation.
    
    The graph has the following flow:
    1. agent: LLM processes scenario and attempts to call a tool
    2. process_decision: Extract decision from tool call
    3. should_continue: Route based on success or remaining retries
    4. retry_logic: If needed, prompt LLM to retry
    
    Returns:
        Compiled LangGraph workflow.
    """
    workflow = StateGraph(GameState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("process_decision", tool_node)
    workflow.add_node("retry_logic", retry_node)

    workflow.set_entry_point("agent")
    
    workflow.add_edge("agent", "process_decision")
    workflow.add_conditional_edges(
        "process_decision",
        should_continue,
        {
            "end_success": END,
            "retry": "retry_logic",
            "end_fail": END
        }
    )
    workflow.add_edge("retry_logic", "agent")
    
    logger.info("LangGraph workflow compiled successfully")
    return workflow.compile()


# --- 4. MAIN EXPERIMENT LOOP ---

def evaluate(
    model_config: Optional[Dict[str, str]] = None,
    input_dir: Optional[str] = None,
    output_file: Optional[str] = None,
) -> None:
    """
    Main evaluation pipeline: Run agents through all scenarios and save results.
    
    Args:
        model_config: Optional dict with 'name' and 'provider' keys selecting
                  which model to evaluate. When None the first entry
                  from EVAL_MODELS is used.
        input_dir: Optional scenario root to evaluate instead of config.SCENARIOS_PATH.
        output_file: Optional CSV path for writing results instead of the default.
    
    Raises:
        IOError: If scenario files cannot be read or results cannot be written.
    """
    global MODEL_PROVIDER, MODEL_NAME, OUTPUT_FILE
    try:
        logger.info("Initializing evaluation pipeline")
        
        # Validate configuration
        config.validate_config()
        
        # Apply model selection
        MODEL_PROVIDER, MODEL_NAME = _get_eval_model_config(model_config)
        OUTPUT_FILE = os.path.join(
            EVALUATIONS_DIR,
            f"results_{_sanitize_filename(MODEL_NAME)}.csv"
        )
        selected_input_dir = input_dir or INPUT_DIR
        selected_output_file = output_file or OUTPUT_FILE
        selected_output_dir = os.path.dirname(selected_output_file) or EVALUATIONS_DIR

        if not os.path.isdir(selected_input_dir):
            raise FileNotFoundError(
                f"Scenario directory not found: {selected_input_dir}. "
                "Generate scenarios first or select an existing scenario directory."
            )
        scenario_batches = []
        for root, dirs, files in os.walk(selected_input_dir):
            # Keep the main benchmark separate from generated sensitivity inputs.
            excluded_dirs = {"prompt_dumps"}
            if input_dir is None:
                excluded_dirs.add("sensitivity_analysis")
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            json_files = sorted(name for name in files if name.endswith(".json"))
            if json_files:
                scenario_batches.append((root, json_files))
        if not scenario_batches:
            raise FileNotFoundError(
                f"No scenario JSON files found in: {selected_input_dir}. "
                "Generate scenarios first or select a directory containing scenarios."
            )

        os.makedirs(EVALUATIONS_DIR, exist_ok=True)
        os.makedirs(selected_output_dir, exist_ok=True)
        
        app = build_graph()
        
        csv_columns = [
            "scenario_id", "game_type", "context", "variation", 
            "scenario_source", "study_name", "generator_prompt_variant",
            "generator_prompt_summary",
            "persona", "model", "action", "action_label", "decision_value",
            "full_args", "reasoning"
        ]
        
        file_exists = os.path.exists(selected_output_file)
        mode = 'a' if file_exists else 'w'
        
        logger.info(f"Starting Evaluation using model: {MODEL_NAME}")
        logger.info(f"Scenario input directory: {selected_input_dir}")
        logger.info(f"Output file: {selected_output_file}")
        
        total_processed = 0
        total_errors = 0
        total_failures = 0

        with open(selected_output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()
                logger.info("Created new results file")

            for root, files in scenario_batches:
                for filename in sorted(files):
                    if not filename.endswith(".json"):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    logger.debug(f"Processing file: {filepath}")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as jf:
                            scenarios = json.load(jf)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {filename}: {e}")
                        total_errors += 1
                        continue
                    except IOError as e:
                        logger.error(f"Cannot read file {filename}: {e}")
                        total_errors += 1
                        continue
                    
                    if TEST_FIRST_N is not None:
                        scenarios = scenarios[:TEST_FIRST_N]

                    logger.info(f"Processing {len(scenarios)} scenarios from {filename}")
                    print(f">>> {filename} ({len(scenarios)} scenarios)")

                    for scenario in scenarios:
                        meta = scenario.get("meta", {})
                        action_labels = meta.get("action_labels", {})
                        # Mapping from presentation tool name to true action
                        option_to_action = meta.get("option_to_action", {})
                        
                        for persona_name, persona_prompt in PERSONAS.items():
                            try:
                                final_state = app.invoke({
                                    "scenario_config": scenario,
                                    "persona_prompt": persona_prompt,
                                    "messages": [],
                                    "retry_count": 0,
                                    "final_decision": None
                                })
                                
                                result = final_state.get("final_decision")
                                
                                if result:
                                    raw_action = result["action"]
                                    # Resolve presentation name back to true
                                    # underlying action via option_to_action.
                                    # Falls back to the raw name when no
                                    # mapping exists (backward compat).
                                    action = option_to_action.get(raw_action, raw_action)
                                    value = result["value"]
                                    reasoning = result["reasoning"]
                                    args_str = json.dumps(result["args"])
                                    logger.debug(f"Scenario {scenario['id']}: {raw_action} -> {action}")
                                else:
                                    action = "FAILED"
                                    value = "N/A"
                                    reasoning = "FAILED TO CALL TOOL"
                                    args_str = "{}"
                                    total_failures += 1
                                    logger.warning(f"No tool call for {scenario['id']}")

                            except Exception as e:
                                logger.error(f"Error on {scenario['id']}: {e}", exc_info=True)
                                action = "ERROR"
                                value = "ERROR"
                                reasoning = str(e)
                                args_str = ""
                                total_errors += 1

                            # Resolve human-readable label for the action
                            action_label = action_labels.get(action, action)

                            # Write Row
                            writer.writerow({
                                "scenario_id": scenario["id"],
                                "game_type": meta.get("game_type", "unknown"),
                                "context": meta.get("context", "unknown"),
                                "variation": meta.get("variation_index", 0),
                                "scenario_source": meta.get("scenario_source", "main"),
                                "study_name": meta.get("study_name", ""),
                                "generator_prompt_variant": meta.get("generator_prompt_variant", ""),
                                "generator_prompt_summary": meta.get("generator_prompt_summary", ""),
                                "persona": persona_name,
                                "model": MODEL_NAME,
                                "action": action,
                                "action_label": action_label,
                                "decision_value": value,  
                                "full_args": args_str,
                                "reasoning": reasoning
                            })
                            
                            total_processed += 1
                            f.flush()
                        
                        print(".", end="", flush=True)
                    print()  # New line after file

        logger.info(f"Evaluation Complete!")
        logger.info(f"Total processed: {total_processed}, Errors: {total_errors}, Failures: {total_failures}")
        logger.info(f"Results saved to: {selected_output_file}")
        print(f"\n--- Evaluation Complete. Results in {selected_output_file} ---")
        
    except Exception as e:
        logger.error(f"Fatal error in evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    evaluate()
