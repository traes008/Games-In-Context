"""
Generator-prompt sensitivity analysis tooling.

This module creates dedicated scenario sets that vary only the generator
prompt wording while keeping game mechanics fixed. It then evaluates models
on those scenarios and writes CSV outputs for downstream analysis.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from . import config
from . import evaluate
from . import generate_scenarios
from . import model

logger = logging.getLogger(__name__)

SENSITIVITY_OUTPUT_DIR = os.path.join(config.SCENARIOS_PATH, "sensitivity_analysis")
SENSITIVITY_RESULTS_DIR = os.path.join(config.EVALUATIONS_PATH, "sensitivity_analysis")


PROMPT_VARIANTS: List[Dict[str, str]] = [
	{
		"name": "baseline",
		"summary": "Original numbered-instruction template (control).",
		"template": "baseline_numbered",
	},
	{
		"name": "block_text",
		"summary": "Single-paragraph block prompt with same constraints.",
		"template": "block_text",
	},
	{
		"name": "no_numbering",
		"summary": "Instructional template without numbered list.",
		"template": "no_numbering",
	},
	{
		"name": "role_card",
		"summary": "Role-card style prompt with compact directives.",
		"template": "role_card",
	},
]


def _sanitize_filename(name: str) -> str:
	"""Sanitize a string for use as a filename on Windows."""
	allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ .")
	cleaned = "".join(ch if ch in allowed else "_" for ch in name)
	cleaned = cleaned.strip().replace(" ", "_")
	return cleaned or "study"


def _resolve_contexts(include_abstract_control: bool) -> List[str]:
	"""Return contexts to use for sensitivity generation."""
	contexts = list(config.SCENARIO_CONTEXTS)
	if include_abstract_control:
		return contexts
	return [ctx for ctx in contexts if ctx != "abstract_game_theory"]


def _build_prompt_variant_map() -> Dict[str, Dict[str, str]]:
	"""Index prompt variants by name for quick lookup."""
	return {variant["name"]: variant for variant in PROMPT_VARIANTS}


def _build_wording_variant_prompt(
	game_title: str,
	rules: str,
	context: str,
	tool_names: List[str],
	swapped: bool,
	template_name: str,
) -> str:
	"""Build wording-variant prompts while preserving main generator logic.

	Core logic is kept consistent with generate_scenarios:
	- k-level vs 2x2 branch
	- strict mechanics fidelity
	- optional reversed option order handling
	- output format constraints
	"""
	klevel = len(tool_names) == 1

	if swapped and len(tool_names) == 2:
		option_a_name = tool_names[1]
		option_b_name = tool_names[0]
	else:
		option_a_name = tool_names[0] if len(tool_names) > 0 else "Option A"
		option_b_name = tool_names[1] if len(tool_names) > 1 else "Option B"

	if klevel:
		number_instruction = (
			"You MUST explicitly preserve all key numeric parameters from the "
			"RULES in the narrative (e.g., target fractions, numeric ranges, "
			"bonus/penalty values, player counts). Do not abstract these into "
			"vague language and do not invent extra numeric targets."
		)
		output_instruction = (
			"End by prompting the user to submit a numeric decision value, "
			"without giving strategy hints."
		)
		instruction_8 = (
			"Frame numeric submission naturally in context (for example: "
			"'place your bid' or 'name your value')."
		)
	else:
		number_instruction = (
			"Do not quote payoff numbers directly. Translate payoff outcomes "
			"into narrative stakes. You must still preserve structural values "
			"that matter for strategy (for example player counts and capacity "
			"thresholds)."
		)
		output_instruction = (
			"End by presenting exactly two options, labeled 'Option A' and "
			"'Option B', without exposing formal action names."
		)
		instruction_8 = (
			"Narratively encode the underlying actions so Option A corresponds "
			f"to {option_a_name} and Option B corresponds to {option_b_name}."
		)

	swap_instruction = ""
	if swapped and not klevel:
		swap_instruction = (
			"The option order is intentionally reversed: the second underlying "
			"action must be described as Option A and the first underlying "
			"action as Option B."
		)

	if template_name == "block_text":
		return (
			f"Write one short scenario in a {context} framing for the game {game_title}. "
			f"Rules to preserve exactly: {rules}. Keep the user as the main decision-maker. "
			f"The strategic mechanics must remain identical to the rules; do not add extra "
			f"actions, hidden information, or side objectives. {number_instruction} "
			f"{output_instruction} {instruction_8} {swap_instruction} "
			"Output only the scenario text (plain text, no JSON)."
		)

	if template_name == "no_numbering":
		return f"""TASK: Write a unique scenario description for this game-theoretic setup.

GAME: {game_title}
RULES: {rules}
CONTEXT: {context}

INSTRUCTIONS:
- Write a short narrative (3-4 sentences).
- Put the user in the central decision role.
- Keep strategic mechanics fully faithful to the RULES.
- {number_instruction}
- Keep context vocabulary natural and domain-appropriate.
- Preserve strategic tension exactly; do not alter incentives.
- {output_instruction}
- {instruction_8}
- {swap_instruction if swap_instruction else 'Keep option order aligned with underlying actions.'}

OUTPUT: Only narrative text. Do not output JSON."""

	if template_name == "role_card":
		return f"""You are writing a scenario card.

Title context: {context}
Game: {game_title}
Fixed rules: {rules}

Card requirements:
Write 3-4 sentences.
User is the decision-maker.
Mechanics and incentives must match the rules exactly.
{number_instruction}
{output_instruction}
{instruction_8}
{swap_instruction if swap_instruction else 'No reversed option order in this instance.'}

Return only plain scenario text, no JSON, no analysis."""

	# baseline_numbered (control)
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
8. {instruction_8}{(' ' + swap_instruction) if swap_instruction else ''}

OUTPUT: Only the narrative text. Do not output JSON."""


def generate_ablation_scenario_text(
	llm: model.LLMWrapper,
	game_title: str,
	rules: str,
	context: str,
	variation_idx: int,
	tool_names: List[str],
	variant: Dict[str, str],
	swapped: bool = False,
) -> str:
	"""Generate one scenario text for a generator-prompt variant arm."""
	if context == "abstract_game_theory":
		return generate_scenarios.generate_scenario_text(
			llm=llm,
			game_title=game_title,
			rules=rules,
			context=context,
			variation_idx=variation_idx,
			tool_names=tool_names,
			swapped=swapped,
		)

	prompt = _build_wording_variant_prompt(
		game_title=game_title,
		rules=rules,
		context=context,
		tool_names=tool_names,
		swapped=swapped,
		template_name=variant.get("template", "baseline_numbered"),
	)
	response = llm.call([HumanMessage(content=prompt)])
	return response.content.strip()


def _transform_tools(template_tools: List[Dict[str, Any]], swapped: bool) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
	"""Rename tools to option labels and capture option->action mapping."""
	if swapped and len(template_tools) == 2:
		ordered_tools = list(reversed(template_tools))
	else:
		ordered_tools = list(template_tools)

	option_to_action: Dict[str, str] = {}
	tools_with_options: List[Dict[str, Any]] = []
	for idx, tool in enumerate(ordered_tools):
		presentation_name = f"action_{'ab'[idx]}"
		option_label = f"Option {'AB'[idx]}"
		option_to_action[presentation_name] = tool["name"]
		tools_with_options.append({
			"name": presentation_name,
			"display_name": option_label,
			"description": f"Select {option_label}.",
			"parameters": tool.get("parameters", {}),
		})

	return tools_with_options, option_to_action


def generate_ablation_scenarios(
	games: Optional[List[str]] = None,
	contexts: Optional[List[str]] = None,
	variations: Optional[int] = None,
	prompt_variants: Optional[List[str]] = None,
	include_abstract_control: bool = False,
) -> None:
	"""Generate sensitivity-analysis scenarios and write them under scenarios/sensitivity_analysis/."""
	try:
		config.validate_config()

		games = games or list(config.SCENARIO_GAMES)
		contexts = contexts or _resolve_contexts(include_abstract_control)
		variations = variations if variations is not None else config.SCENARIO_VARIATIONS_PER_CONTEXT
		variant_map = _build_prompt_variant_map()
		selected_variants = [variant_map[name] for name in (prompt_variants or list(variant_map.keys()))]

		llm = model.LLMFactory.get(
			config.SCENARIO_GENERATOR_PROVIDER,
			config.SCENARIO_GENERATOR_MODEL,
			temperature=config.TEMPERATURE_GENERATION,
		)

		os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)
		total_scenarios = 0

		for game_name in games:
			template = generate_scenarios.load_template(game_name)
			game_type = template.get("game_type", game_name)
			target_dir = os.path.join(SENSITIVITY_OUTPUT_DIR, game_type)
			os.makedirs(target_dir, exist_ok=True)

			logger.info("Generating sensitivity scenarios for %s", game_name)

			for context in contexts:
				for variant in selected_variants:
					output_name = f"{game_name}_{context}_{variant['name']}.json"
					output_filepath = os.path.join(target_dir, _sanitize_filename(output_name))
					scenario_batch: List[Dict[str, Any]] = []

					logger.info(
						"Generating %s scenarios for %s (%s, %s)",
						variations,
						game_name,
						context,
						variant["name"],
					)

					for i in range(variations):
						swapped = len(template.get("tools", [])) == 2 and (i % 2 == 1)
						tool_names = [tool["name"] for tool in template.get("tools", [])]
						description = generate_ablation_scenario_text(
							llm=llm,
							game_title=template["title"],
							rules=template.get("rules", "Standard rules apply."),
							context=context,
							variation_idx=i,
							tool_names=tool_names,
							variant=variant,
							swapped=swapped,
						)

						tools_with_options, option_to_action = _transform_tools(
							template.get("tools", []),
							swapped,
						)

						scenario_batch.append({
							"id": f"ablation_{game_name}_{context}_{variant['name']}_var_{i}",
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
								"scenario_source": "sensitivity_analysis",
								"study_name": "generator_prompt_sensitivity",
								"generator_prompt_variant": variant["name"],
								"generator_prompt_summary": variant["summary"],
							},
						})
						total_scenarios += 1

					with open(output_filepath, "w", encoding="utf-8") as f:
						json.dump(scenario_batch, f, indent=2)

					logger.info("Saved %s scenarios to %s", len(scenario_batch), output_filepath)

		logger.info("Sensitivity generation complete. Total scenarios generated: %s", total_scenarios)

	except Exception as e:
		logger.error("Fatal error in sensitivity scenario generation: %s", e, exc_info=True)
		sys.exit(1)


def evaluate_ablation(model_names: Optional[List[str]] = None) -> None:
	"""Evaluate configured models only on the sensitivity-analysis scenario tree."""
	os.makedirs(SENSITIVITY_RESULTS_DIR, exist_ok=True)

	for model_config in config.get_model_configs(model_names):
		output_file = os.path.join(
			SENSITIVITY_RESULTS_DIR,
			f"results_{_sanitize_filename(model_config['name'])}_ablation.csv",
		)
		logger.info(
			"Evaluating sensitivity scenarios for model %s (%s)",
			model_config["name"],
			model_config["provider"],
		)
		evaluate.evaluate(
			model_config=model_config,
			input_dir=SENSITIVITY_OUTPUT_DIR,
			output_file=output_file,
		)


def main() -> None:
	"""CLI entrypoint for generating and evaluating sensitivity scenarios."""
	parser = argparse.ArgumentParser(description="Generator-prompt sensitivity analysis")
	parser.add_argument("--generate", action="store_true", help="Generate sensitivity scenarios")
	parser.add_argument("--evaluate", action="store_true", help="Evaluate sensitivity scenarios")
	parser.add_argument(
		"--models",
		nargs="+",
		metavar="MODEL",
		help="Model name(s) to evaluate, matching src.config.EVAL_MODELS",
	)
	parser.add_argument(
		"--include-abstract-control",
		action="store_true",
		help="Also generate abstract_game_theory control scenarios for each prompt arm",
	)
	parser.add_argument(
		"--games",
		nargs="+",
		metavar="GAME",
		help="Optional game names to include (e.g. prisoners_dilemma chicken)",
	)
	parser.add_argument(
		"--contexts",
		nargs="+",
		metavar="CONTEXT",
		help="Optional contexts to include (e.g. war business game_show)",
	)
	parser.add_argument(
		"--variants",
		nargs="+",
		metavar="VARIANT",
		help="Optional prompt variants to include (baseline block_text no_numbering role_card)",
	)
	parser.add_argument(
		"--variations",
		type=int,
		help="Optional number of scenario variations per game/context/variant",
	)
	args = parser.parse_args()

	run_any = any([args.generate, args.evaluate])
	run_generate = args.generate or (not run_any)
	run_evaluate = args.evaluate

	if run_generate:
		generate_ablation_scenarios(
			games=args.games,
			contexts=args.contexts,
			variations=args.variations,
			prompt_variants=args.variants,
			include_abstract_control=args.include_abstract_control,
		)
	if run_evaluate:
		evaluate_ablation(model_names=args.models)


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	)
	main()
