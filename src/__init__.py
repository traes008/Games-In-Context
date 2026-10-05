"""
Game Theoretic Benchmark - LLM Decision-Making Evaluation Framework.

This package provides tools for generating game-theoretic scenarios and evaluating
how different LLMs make strategic decisions under various contextual settings.

Main modules:
- config: Configuration management for all experiment parameters
- model: LLM factory for unified interface across multiple providers
- generate_scenarios: Create contextual game scenarios via LLM prompting
- sensitivity_analysis: Generate prompt-ablation scenario sets and evaluate them
- evaluate: Run LLM agents through game-theoretic scenarios
- analyse: Analyze and visualize evaluation results
- regression_analysis: Statistical analysis of evaluation results
"""

__version__ = "0.1.0"
__license__ = "MIT"

import logging

# Configure package-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
	"config",
	"model",
	"generate_scenarios",
	"sensitivity_analysis",
	"evaluate",
	"analyse",
	"regression_analysis",
]
