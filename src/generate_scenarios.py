"""
Scenario generation module for creating contextual game-theoretic scenarios.

This module generates variations of classical game-theoretic scenarios across different
narrative contexts by leveraging LLM prompt-based generation. It creates scenarios that
maintain the same strategic structure while varying narrative presentation.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage

from . import config
from . import model

# Configure module logger
logger = logging.getLogger(__name__)

# Configuration Paths
TEMPLATES_DIR = "templates"
OUTPUT_DIR = "scenarios"


def load_template(game_name: str) -> Dict[str, Any]:
    """
    Load a game template from JSON.
    
    Args:
        game_name: Name of the game (e.g., 'prisoners_dilemma').
        
    Returns:
        Loaded template dictionary.
        
    Raises:
        FileNotFoundError: If template file doesn't exist.
        json.JSONDecodeError: If template is invalid JSON.
    """
    path = os.path.join(TEMPLATES_DIR, f"{game_name}.json")
    if not os.path.exists(path):
        logger.error(f"Template not found at {path}")
        raise FileNotFoundError(f"Template not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            template = json.load(f)
            logger.debug(f"Loaded template for {game_name}")
            return template
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in template {path}: {e}")
        raise


def generate_scenario_text(
    llm: model.LLMWrapper,
    game_title: str,
    rules: str,
    context: str,
    variation_idx: int,
    tool_names: List[str],
    swapped: bool = False
) -> str:
    """
    Generate a contextual scenario description using an LLM.

    Splits logic between **2x2 games** (two discrete options) and **k-level
    reasoning games** (single numeric submission) so each game type gets
    tailored instructions.

    For the abstract_game_theory context, returns static variations to ensure
    a pure control group without LLM generation artifacts.

    When swapped=True (only affects 2-tool games), the mapping between option
    labels and underlying actions is reversed: Option A -> action_b,
    Option B -> action_a.  This controls for positional bias in LLM evaluation.

    Args:
        llm: LLM instance for text generation.
        game_title: Title of the game.
        rules: Game rules in structured format.
        context: Domain context (e.g., 'war', 'business', 'game_show').
        variation_idx: Index of this variation.
        tool_names: List of tool/action names to weave into narrative.
        swapped: If True, reverse the option label mapping for 2-tool games.

    Returns:
        Generated scenario description text.
    """
    klevel = len(tool_names) == 1

    if swapped and len(tool_names) == 2:
        option_a_name = tool_names[1]  # action_b displayed as Option A
        option_b_name = tool_names[0]  # action_a displayed as Option B
    else:
        option_a_name = tool_names[0] if len(tool_names) > 0 else "Option A"
        option_b_name = tool_names[1] if len(tool_names) > 1 else "Option B"

    # --- CONTROL GROUP BYPASS ---
    # For abstract game theory, use static control descriptions
    if context == "abstract_game_theory":
        # When swapped, reverse Action_A/Action_B labels in the rules text
        # so that the mapping stays consistent with the presentation order
        # (action_a tool = Option A, action_b tool = Option B).
        display_rules = rules
        if swapped and not klevel:
            display_rules = (
                rules
                .replace("Action_A", "_ACTION_SWAP_TEMP_")
                .replace("Action_B", "Action_A")
                .replace("_ACTION_SWAP_TEMP_", "Action_B")
            )

        if klevel:
            variations = [
                f"You are an agent playing a game called {game_title}. {display_rules} Please submit your numeric decision.",
                f"Consider the following game: {game_title}. The rules are: {display_rules} What is your optimal numeric choice?",
                f"You are participating in a strategic interaction ({game_title}). {display_rules} Submit your chosen value.",
            ]
        else:
            variations = [
                f"You are Player A. You are playing a game called {game_title}. {display_rules} Choose between Option A or Option B.",
                f"Consider the following game: {game_title}. The rules are: {display_rules} Your choices are Option A or Option B.",
                f"You are participating in a strategic interaction ({game_title}). {display_rules} Decide: Option A or Option B?",
            ]
        return variations[variation_idx % len(variations)]

    # --- EXPERIMENTAL GROUP (Narrative Generation) ---
    prompt = build_generation_prompt(
        game_title=game_title,
        rules=rules,
        context=context,
        tool_names=tool_names,
        swapped=swapped,
    )

    try:
        logger.debug(f"Generating scenario for {game_title} in {context} context")
        response = llm.call([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating scenario for {game_title} ({context}): {e}")
        return f"Error generating scenario. Context: {context}. Game: {game_title}."


def build_generation_prompt(
    game_title: str,
    rules: str,
    context: str,
    tool_names: List[str],
    swapped: bool = False,
    style_instruction: Optional[str] = None,
    focus_instruction: Optional[str] = None,
) -> str:
    """Build the standard scenario-generation prompt with optional style overrides.

    This is the single source of truth for the generator prompt template used
    by both the main scenario generator and sensitivity-analysis variants.

    When style_instruction and focus_instruction are omitted, the prompt text
    matches the default production generator prompt.
    """
    klevel = len(tool_names) == 1

    if swapped and len(tool_names) == 2:
        option_a_name = tool_names[1]  # action_b displayed as Option A
        option_b_name = tool_names[0]  # action_a displayed as Option B
    else:
        option_a_name = tool_names[0] if len(tool_names) > 0 else "Option A"
        option_b_name = tool_names[1] if len(tool_names) > 1 else "Option B"

    if klevel:
        number_instruction = (
            "You MUST explicitly preserve all key numeric parameters from the "
            "RULES in the narrative (e.g., target fractions, numeric ranges, "
            "bonus/penalty values, player counts). DO NOT abstract these into "
            "vague language and DO NOT introduce any numeric relationships or "
            "mathematical targets that are not present in the RULES - the "
            "exact values are required for the scenario to be solvable."
        )
        output_instruction = (
            "End by prompting the user to submit their chosen numeric value "
            "for the game, without revealing the strategic reasoning behind "
            "the optimal choice."
        )
        instruction_8 = (
            "Frame the numeric submission naturally within the narrative "
            "context (e.g., 'place your bid', 'name your figure')."
        )
    else:
        number_instruction = (
            "DO NOT mention the mathematical payoff numbers or point values "
            "explicitly. Instead, translate payoff outcomes into narrative "
            "stakes. However, you MUST explicitly state any structural game "
            "parameters such as the exact number of players/agents and "
            "capacity thresholds (e.g., '60 out of 100'), as these are "
            "essential for strategic reasoning."
        )
        output_instruction = (
            "End by presenting two options as 'Option A' and 'Option B' "
            "without revealing the underlying action names."
        )
        instruction_8 = (
            "Weave the action semantics of each option naturally into the "
            "narrative context so the reader understands the choice without "
            "seeing the formal game-theoretic labels."
        )

    swap_instruction = ""
    if swapped and not klevel:
        swap_instruction = (
            "\n9. CRITICAL - REVERSED OPTION ORDER: Present the SECOND action "
            "from the RULES (Action_B / the second strategic choice) as 'Option A', "
            "and the FIRST action (Action_A / the first strategic choice) as 'Option B'. "
            "The narrative descriptions of each option must reflect this reversed mapping."
        )

    extra_style = ""
    if style_instruction:
        extra_style += f"\n9. STYLE OVERRIDE: {style_instruction}"
    if focus_instruction:
        extra_style += f"\n10. FOCUS OVERRIDE: {focus_instruction}"

    return f"""TASK: Write a unique scenario description for a specific game theory situation.

GAME: {game_title}
RULES: {rules}
CONTEXT SETTING: {context}

INSTRUCTIONS:
1. Write a short, narrative (3-4 sentences) describing this specific situation.
2. The User is in the role of the primary decision-maker in this context.
3. The narrative must perfectly map to the exact mechanics of the RULES provided, including the relationship between the players (e.g., portray allies/partners for coordination games, rivals for competitive games).
4. {number_instruction}
5. Make this variation distinct. Use specific vocabulary relevant to the {context} setting.
6. The underlying strategic tension must remain identical to the original RULES.
7. {output_instruction}
8. {instruction_8}{swap_instruction}{extra_style}

OUTPUT: Only the narrative text. Do not output JSON."""


def main() -> None:
    """Main scenario generation pipeline."""
    try:
        logger.info("Starting scenario generation pipeline")
        
        # Validate configuration
        config.validate_config()
        
        # 1. Load Configuration from config.py
        games = config.SCENARIO_GAMES
        contexts = config.SCENARIO_CONTEXTS
        variations = config.SCENARIO_VARIATIONS_PER_CONTEXT
        
        # 2. Initialize LLM (Generator)
        provider = config.SCENARIO_GENERATOR_PROVIDER
        model_name = config.SCENARIO_GENERATOR_MODEL
        logger.info(f"Initializing Generator LLM: {provider} / {model_name}")
        
        try:
            llm = model.LLMFactory.get(provider, model_name, temperature=config.TEMPERATURE_GENERATION)
        except ValueError as e:
            logger.error(f"Failed to initialize LLM: {e}")
            sys.exit(1)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 3. Main Generation Loop
        total_scenarios = 0
        for game_name in games:
            logger.info(f"Processing Game: {game_name}")
            
            try:
                template = load_template(game_name)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping {game_name}: {e}")
                continue
                
            game_type = template.get("game_type", game_name)
            
            for context in contexts:
                target_dir = os.path.join(OUTPUT_DIR, game_type)
                os.makedirs(target_dir, exist_ok=True)
                output_filepath = os.path.join(target_dir, f"{game_name}_{context}.json")

                logger.info(f"Generating {variations} scenarios for {game_name} ({context})")

                scenario_batch = []

                for i in range(variations):
                    try:
                        # Alternate option order to control for positional bias
                        swapped = (i % 2 == 1)
                        
                        # Get tool names
                        tool_names = [tool["name"] for tool in template.get("tools", [])]
                        
                        # Generate narrative
                        description = generate_scenario_text(
                            llm,
                            template["title"],
                            template.get("rules", "Standard rules apply."),
                            context,
                            i,
                            tool_names,
                            swapped=swapped,
                        )
                        
                        # Transform tools
                        # When swapped, reverse tool presentation order so
                        # the second action's semantics become Option A.
                        # IMPORTANT: Always name tools action_a / action_b in
                        # presentation order so the LLM cannot infer the
                        # mapping from the tool name.  A separate
                        # option_to_action mapping records the true identity.
                        template_tools = template.get("tools", [])
                        if swapped and len(template_tools) == 2:
                            ordered_tools = list(reversed(template_tools))
                        else:
                            ordered_tools = list(template_tools)
                        
                        # Build the mapping: presentation name -> original action
                        option_to_action = {}
                        tools_with_options = []
                        for idx, tool in enumerate(ordered_tools):
                            presentation_name = f"action_{'ab'[idx]}"
                            option_label = f"Option {'AB'[idx]}"
                            option_to_action[presentation_name] = tool["name"]
                            transformed_tool = {
                                "name": presentation_name,
                                "display_name": option_label,
                                "description": f"Select {option_label}.",
                                "parameters": tool.get("parameters", {}),
                            }
                            tools_with_options.append(transformed_tool)
                        
                        # Construct scenario
                        scenario_obj = {
                            "id": f"{game_name}_{context}_var_{i}",
                            "description": description,
                            "tools": tools_with_options,
                            "meta": {
                                "game_type": game_type,
                                "game_title": template["title"],
                                "context": context,
                                "variation_index": i,
                                "rules_ref": template.get("rules", ""),
                                "option_order_swapped": swapped,
                                "action_labels": template.get("action_labels", {}),
                                "option_to_action": option_to_action,
                            },
                        }
                        
                        scenario_batch.append(scenario_obj)
                        total_scenarios += 1
                        print(f"  [{i+1}/{variations}] Generated variation", end="\r")
                        
                    except Exception as e:
                        logger.error(f"Error generating variation {i} for {game_name} ({context}): {e}")
                        continue

                # Save batch to file
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(scenario_batch, f, indent=2)
                    logger.info(f"Saved {len(scenario_batch)} scenarios to {output_filepath}")
                except IOError as e:
                    logger.error(f"Failed to save scenarios to {output_filepath}: {e}")
                    continue

                print()  # New line after progress

        logger.info(f"Generation Complete. Total scenarios generated: {total_scenarios}")
        
    except Exception as e:
        logger.error(f"Fatal error in scenario generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()

