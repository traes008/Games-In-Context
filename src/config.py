"""Configuration module for Game Theoretic Benchmark."""

import logging
from typing import Dict, List, Optional

# Configure module logger
logger = logging.getLogger(__name__)

# --- PATHS ---
SCENARIOS_PATH: str = "scenarios/"
RESULTS_PATH: str = "results/"
EVALUATIONS_PATH: str = "results/evaluations/"

# --- SCENARIO GENERATION ---
SCENARIO_GENERATOR_PROVIDER: str = "google"
SCENARIO_GENERATOR_MODEL: str = "gemini-2.5-flash"
SCENARIO_VARIATIONS_PER_CONTEXT: int = 30

# --- SCENARIO DEFINITIONS ---
SCENARIO_GAMES: List[str] = [
    "prisoners_dilemma",
    "chicken",
    "beauty_contest",
    "travelers_dilemma",
    "stag_hunt",
    "el_farol_bar"
]
SCENARIO_CONTEXTS: List[str] = [
    "abstract_game_theory",
    "war",
    "business",
    "game_show",
    "family fight",
]

# --- EVALUATION ---
EVAL_MODELS: List[Dict[str, str]] = [
    # --- Llama ---
    # {"name": "llama3.1:8b", "provider": "ollama"},
    # {"name": "llama3.1:70b", "provider": "ollama"},
    # {"name": "llama3.2:1b", "provider": "ollama"},
    # {"name": "llama3.2:3b", "provider": "ollama"},
    # {"name": "llama3.3:70b", "provider": "ollama"},
    # --- DeepSeek ---
    # {"name": "deepseek-r1:7b", "provider": "ollama"},
    # {"name": "deepseek-r1:14b", "provider": "ollama"},
    # {"name": "deepseek-r1:32b", "provider": "ollama"},
    # {"name": "deepseek-r1:70b", "provider": "ollama"},
    # {"name": "deepseek-v2.5:236b", "provider": "ollama"},
    # --- Qwen ---
    # {"name": "qwen3:8b", "provider": "ollama"},
    # {"name": "qwen3:14b", "provider": "ollama"},
    {"name": "qwen3:32b", "provider": "ollama"},
    # {"name": "qwen2.5:7b", "provider": "ollama"},
    # {"name": "qwen2.5:14b", "provider": "ollama"},
    # {"name": "qwen2.5:32b", "provider": "ollama"},
    # {"name": "qwen2.5:72b", "provider": "ollama"},
    # --- Gemma ---
    {"name": "gemma3:4b", "provider": "ollama"},
    # {"name": "gemma3:12b", "provider": "ollama"},
    # {"name": "gemma3:27b", "provider": "ollama"},
    # --- Phi ---
    {"name": "phi4:14b", "provider": "ollama"},
    # --- Mistral ---
    # {"name": "mistral:7b", "provider": "ollama"},
    {"name": "mistral-small:24b", "provider": "ollama"},
    # --- Gemini (Google API) ---
    {"name": "gemini-2.5-flash", "provider": "google"},
]

# --- PERSONAS ---
PERSONAS: Dict[str, str] = {
    "generic": "You are a helpful assistant.",
    "rational": "You are a cold, rational assistant. Your objective is to maximize your own payoff.",
    "emotional": "You are an empathetic assistant.",
}

# --- DEBUGGING ---
TEST_FIRST_N: Optional[int] = 2  # For debugging: only test first N scenarios, if None then test all

# --- RETRY POLICY ---
MAX_RETRY_COUNT: int = 2  # Maximum retries for LLM tool invocation failures

# --- TEMPERATURE SETTINGS ---
TEMPERATURE_GENERATION: float = 0.7  # Temperature for scenario generation (higher = more creative)
TEMPERATURE_EVALUATION: float = 0.0  # Temperature for evaluation (0 = deterministic)


def get_model_configs(names: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Return model configurations filtered by name.

    Args:
        names: Optional list of model names to select. If None, returns all
               models from EVAL_MODELS.

    Returns:
        List of model config dicts with 'name' and 'provider' keys.

    Raises:
        ValueError: If a requested model name is not found in EVAL_MODELS.
    """
    if names is None:
        return list(EVAL_MODELS)

    models_by_name = {m["name"]: m for m in EVAL_MODELS}
    selected: List[Dict[str, str]] = []
    for n in names:
        if n not in models_by_name:
            available = ", ".join(models_by_name.keys())
            raise ValueError(
                f"Model '{n}' not found in EVAL_MODELS. Available: {available}"
            )
        selected.append(models_by_name[n])
    return selected


def validate_config() -> bool:
    """
    Validate configuration consistency and required fields.
    
    Returns:
        bool: True if all validations pass.
        
    Raises:
        ValueError: If any critical configuration is missing or invalid.
    """
    errors = []
    
    # Validate generator configuration
    if not SCENARIO_GENERATOR_MODEL:
        errors.append("SCENARIO_GENERATOR_MODEL is not configured")
    if not SCENARIO_GENERATOR_PROVIDER:
        errors.append("SCENARIO_GENERATOR_PROVIDER is not configured")
    
    # Validate evaluation configuration
    if not EVAL_MODELS or len(EVAL_MODELS) == 0:
        errors.append("EVAL_MODELS is empty or not configured")
    for model in EVAL_MODELS:
        if not model.get("name") or not model.get("provider"):
            errors.append(f"Invalid model config: {model}")
    
    # Validate personas
    if not PERSONAS or len(PERSONAS) == 0:
        errors.append("PERSONAS is empty or not configured")
    
    # Validate scenarios
    if not SCENARIO_GAMES or len(SCENARIO_GAMES) == 0:
        errors.append("SCENARIO_GAMES is empty or not configured")
    if not SCENARIO_CONTEXTS or len(SCENARIO_CONTEXTS) == 0:
        errors.append("SCENARIO_CONTEXTS is empty or not configured")
    
    # Validate paths
    if not SCENARIOS_PATH:
        errors.append("SCENARIOS_PATH is not configured")
    if not RESULTS_PATH:
        errors.append("RESULTS_PATH is not configured")
    
    if errors:
        error_msg = "Configuration validation failed:\n- " + "\n- ".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")
    return True
