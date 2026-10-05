"""
Main entry point for the Game Theoretic Benchmark pipeline.

Run this script to execute the full pipeline:
    1. Generate scenarios from templates
    2. Evaluate models on scenarios
    3. Analyze and visualize results

Usage:
    python main.py                                  # Run full pipeline (all models)
    python main.py --generate                       # Generate scenarios only
    python main.py --evaluate                       # Evaluate all models in EVAL_MODELS
    python main.py --evaluate --models qwen3:32b   # Evaluate specific model(s)
    python main.py --evaluate --limit 1           # One scenario per input file
    python main.py --evaluate --all-scenarios     # Evaluate every scenario
    python main.py --analyze                        # Analyze results only
    python main.py --regress                        # Run advanced regression analysis
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Execute the full game theoretic benchmark pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Game Theoretic Benchmark Pipeline"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate scenarios from templates"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models on scenarios"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze and visualize results"
    )
    parser.add_argument(
        "--regress",
        action="store_true",
        help="Run advanced regression/statistical analysis"
    )
    parser.add_argument(
        "--generate-ablation",
        action="store_true",
        help="Generate generator-prompt sensitivity analysis scenarios"
    )
    parser.add_argument(
        "--evaluate-ablation",
        action="store_true",
        help="Evaluate models only on sensitivity-analysis scenarios"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Model name(s) to evaluate, as defined in config.py EVAL_MODELS "
             "(e.g. --models qwen3:32b gemini-2.5-flash). "
             "If omitted, all models in EVAL_MODELS are used."
    )
    scenario_limit = parser.add_mutually_exclusive_group()
    scenario_limit.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Evaluate the first N scenarios in each input file (N must be positive)"
    )
    scenario_limit.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Evaluate every scenario, overriding config.py TEST_FIRST_N"
    )
    
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    
    any_stage_requested = any([
        args.generate,
        args.evaluate,
        args.analyze,
        args.regress,
        args.generate_ablation,
        args.evaluate_ablation,
    ])

    # If no specific task specified, run the original full pipeline.
    run_generate = args.generate or (not any_stage_requested)
    run_evaluate = args.evaluate or (not any_stage_requested)
    run_analyze = args.analyze or (not any_stage_requested)
    # Combine advanced regression with the analysis stage by default.
    # `--regress` still allows running it standalone.
    run_regress = args.regress or run_analyze
    
    try:
        # Resolve model selection once (used by --evaluate)
        from src import config as cfg
        if args.all_scenarios:
            cfg.TEST_FIRST_N = None
        elif args.limit is not None:
            cfg.TEST_FIRST_N = args.limit
        model_configs = cfg.get_model_configs(args.models)
        
        if run_generate:
            logger.info("=" * 50)
            logger.info("STAGE 1: Generating Scenarios")
            logger.info("=" * 50)
            from src import generate_scenarios
            generate_scenarios.main()
        
        if run_evaluate:
            logger.info("=" * 50)
            logger.info("STAGE 2: Evaluating Models")
            logger.info("=" * 50)
            from src import evaluate
            for mc in model_configs:
                logger.info(f"Evaluating model: {mc['name']} ({mc['provider']})")
                evaluate.evaluate(model_config=mc)

        if args.generate_ablation:
            logger.info("=" * 50)
            logger.info("ABLATION: Generating Scenarios")
            logger.info("=" * 50)
            from src import sensitivity_analysis
            sensitivity_analysis.generate_ablation_scenarios()

        if args.evaluate_ablation:
            logger.info("=" * 50)
            logger.info("ABLATION: Evaluating Models")
            logger.info("=" * 50)
            from src import sensitivity_analysis
            sensitivity_analysis.evaluate_ablation(model_names=args.models)
        
        if run_analyze:
            logger.info("=" * 50)
            logger.info("STAGE 3: Analyzing Results")
            logger.info("=" * 50)
            from src import analyse
            analyse.analyse()

        if run_regress:
            logger.info("=" * 50)
            logger.info("STAGE 4: Regression Analysis")
            logger.info("=" * 50)
            from src import regression_analysis
            regression_analysis.run_regression_analysis()
        
        logger.info("=" * 50)
        logger.info("Pipeline Complete!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
