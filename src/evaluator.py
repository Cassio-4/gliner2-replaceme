"""Model evaluation module for assessing GLiNER2 NER performance.

This module evaluates a (possibly pruned) GLiNER2 model with exact-span
precision/recall/F1 on a LeNER-Br or UlyssesNER-Br CoNLL split.
"""

import argparse
import logging
from typing import Optional
import yaml
from colorama import Fore, init

from .utils import eval_model, seed_all

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=(
        f"{Fore.CYAN}%(asctime)s "
        f"{Fore.YELLOW}[%(levelname)s] "
        f"{Fore.RESET}%(message)s"
    ),
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

seed_all()


def evaluator(
    model_path: str,
    dataset: str,
    eval_subset: str = "test",
    threshold: float = 0.5,
    batch_size: int = 16,
    eval_size: Optional[int] = None,
) -> dict:
    """Evaluate a GLiNER2 model on a legal NER dataset.

    Args:
        model_path: Path to GLiNER2 model (e.g. the pruned ReplaceMe output)
        dataset: Dataset to evaluate on ("lener" or "ulysses")
        eval_subset: CoNLL split to score (train/dev/test)
        threshold: Entity confidence threshold passed to batch_extract_entities
        batch_size: Inference batch size
        eval_size: Optional limit on the number of sentences

    Returns:
        Dictionary with per-entity, micro and macro precision/recall/F1
    """
    logging.info(f"{Fore.GREEN}Evaluating {model_path} on the {eval_subset} split of {dataset}{Fore.RESET}")
    results = eval_model(model_path, dataset, eval_subset, threshold, batch_size, eval_size)
    logging.info(
        f"{Fore.GREEN}Evaluation completed: micro F1 = {results['micro']['f1']:.4f}{Fore.RESET}"
    )
    return results


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"{Fore.RED}Config file not found: {config_path}{Fore.RESET}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"{Fore.RED}Invalid YAML in config file: {config_path}{Fore.RESET}")
        raise


def run_from_config() -> None:
    """Run model evaluation from configuration file.
    
    Reads evaluation parameters from a YAML config file and executes the evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Run model evaluation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    
    try:
        config = read_config(args.config)
        seed_all(config.pop("seed", 42))
        evaluator(**config)
    except Exception as e:
        logging.error(f"{Fore.RED}Evaluation failed: {str(e)}{Fore.RESET}")
        raise