import argparse
import glob
import inspect
import logging
import os

import torch
import yaml
from colorama import Fore, Style, init

# Import local modules
from .cosine_dist import cosine_dist
from .distance import profile_distances
from .evaluator import evaluator
from .lstsq import lstsq
from .utils import seed_all, select_non_overlapping_blocks

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages with timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def save_experiment_config(config, model_dir):
    # Not "config.json": the model's own config already uses that name
    config_path = os.path.join(model_dir, "pipeline_config.yaml")
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logging.info(f"Saved experiment config to {config_path}")

def ReplaceMe_pipeline(config):
    # Snapshot before distances_path is overwritten below, so the saved config matches the input
    experiment_config = dict(config)
    # Extract the relevant parameters based on function signatures
    signature = inspect.signature(profile_distances)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    if config['distances_path'] is None:
        # Profile distances using filtered configuration
        profile_distances(**filtered_config)
        config['distances_path'] = "./distances.pth"
    # Determine method and apply configurations accordingly
    if config["method"] == "lstsq":
        signature = inspect.signature(lstsq)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        path = lstsq(**filtered_config)
    else:
        signature = inspect.signature(cosine_dist)
        filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
        
        # Load average distances and select non-overlapping blocks
        average_distances = torch.load(filtered_config['distances_path'])  
        selected_blocks = select_non_overlapping_blocks(
            average_distances, 
            filtered_config['layers_to_skip'], 
            num_blocks=filtered_config['num_A'], 
            merge_consecutive=filtered_config['merge_consecutive']
        )
        
        # Calculate start and end IDs, and number of layers
        start_ids = sorted([x[0] for x in selected_blocks])
        end_ids = sorted([x[1] for x in selected_blocks])
        num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
        num_layers = [sum(num_layers[:i]) for i in range(len(start_ids) + 1)]
        
        # Iterate over each selected block
        for i in range(len(selected_blocks)):
            path = cosine_dist(**filtered_config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i])
            filtered_config["model_path"] = path
    
    # Save before evaluating, so the config is kept even if evaluation fails
    save_experiment_config(experiment_config, path)

    # The model was just rewritten, so results from an earlier run in this directory are stale
    # and eval_model would otherwise return them as cached
    for stale_result in glob.glob(os.path.join(glob.escape(path), "benchmark_*.json")):
        os.remove(stale_result)

    # Evaluate using the updated configuration
    signature = inspect.signature(evaluator)
    filtered_config = {k: v for k, v in config.items() if k in signature.parameters}
    filtered_config["model_path"] = path
    evaluator(**filtered_config)

def read_config(config_path: str):
    # Read and return the YAML configuration
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_from_config():
    # Parse command-line arguments for configuration file path
    parser = argparse.ArgumentParser(
        description="Run LSTSQ for linear transform estimation based on a configuration file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    
    # Execute pipeline based on parsed configuration
    args = parser.parse_args()
    config = read_config(args.config)
    # Keep the seed in the config, including the default, so it gets saved with the experiment
    seed_all(config.setdefault("seed", 42))
    ReplaceMe_pipeline(config)

def main() -> None:
    run_from_config()

if __name__ == "__main__":
    main()
