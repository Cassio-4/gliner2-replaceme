import argparse
import gc
import logging
import os
from typing import Optional
import torch
import yaml     
from colorama import Fore, init
from gliner2 import AutoExtractor
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from .utils import (adam_method, get_gliner2_calib_dataloader, optimizing_method,
                    select_non_overlapping_blocks, truncate_model, seed_all)

# Initialize colorama for Windows compatibility
init(autoreset=True)

# Configure logging to display colored messages and timestamps
logging.basicConfig(
    format=f'{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

seed_all()


def cosine_dist(
    model_path: str,
    dataset: str,
    batch_size: int,
    max_length: int,
    layers_to_skip: int,
    dataset_size: Optional[int] = None,
    dataset_subset: Optional[str] = "dev",
    activations_save_path: Optional[str] = None,
    use_4bit: bool = False,
    save_path: Optional[str] = None,
    min_distance_layer: Optional[int] = None,
    token: Optional[str] = None,
    save_transform_only: bool = False,
    diag: bool = False,
    loss: str = "cosine",
    solver: str = "adam",
    thri: bool = False,
    two_vectors: bool = False,
    start_id: int = 0,
    end_id: int = 0,
    num_layer: int = 0,
    distances_path: str = "./distances.pth",
    num_A: int = 1,
    merge_consecutive: bool = True,
    accurate: bool= False
) -> str:
    """Calculate cosine distance between model layers and apply transformations.
    
    Args:
        model_path: Path to pretrained model
        dataset: Name of dataset to use
        dataset_column: Column in dataset containing text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        layers_to_skip: Number of layers to skip
        dataset_size: Optional size limit for dataset
        dataset_subset: Subset of dataset to use (train/eval)
        activations_save_path: Path to save activations
        use_4bit: Whether to use 4-bit quantization
        save_path: Path to save transformed model
        min_distance_layer: index of start layer for cut
        token: Authentication token
        save_transform_only: Whether to only save the transform
        diag: Whether to use diagonal matrix
        loss: Loss function type
        solver: Optimization solver type
        thri: Whether to use three vectors
        two_vectors: Whether to use two vectors
        start_id: Starting layer ID
        end_id: Ending layer ID
        num_layer: Number of layers
        distances_path: Path to save distance metrics
        num_A: Number of LT transforms
        merge_consecutive: Whether to merge consecutive LT transforms
        accurate: Whether to fit cosine(xA + z, y) instead of cosine(xA, y - z)

    Returns:
        Path where transformed model is saved
    """
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Load model and tokenizer
    # Load GLiNER2 model and profile its HF encoder (e.g. DebertaV2Model)
    extractor = AutoExtractor.from_pretrained(model_path)
    extractor.to("cuda" if torch.cuda.is_available() else "cpu")
    extractor.eval()

    model = extractor.encoder
    model.config.output_hidden_states = True
    model.eval()
    
    dataloader = get_gliner2_calib_dataloader(
        extractor,
        dataset,
        dataset_subset,
        dataset_size,
        batch_size,
        max_len=max_length
        )
    
     # Setup activation hooks
    def save_mlp_activation(name: str):
        """Returns a hook function that saves module outputs."""
        def hook(module, input, output):
            mlp_activations[name] = output.detach()
        return hook
    
    hooks = []
    mlp_activations = {}
    # DebertaV2Layer: attention -> intermediate (dense+GELU) -> output (dense -> dropout -> LayerNorm(x + residual)).
    # output.dense is the FFN down-projection, i.e. the counterpart of Llama's mlp.down_proj (pre-residual, pre-LN).
    # Per layer k:  A_k = attention.output(...)            (already LayerNorm'd, the residual into the FFN block)
    #               M_k = output.dense(GELU(W1 A_k))        <- this hook captures M_k
    #               h_k = output.LayerNorm(M_k + A_k)       <- what outputs.hidden_states[k + 1] holds
    # A_k itself is never captured, and neither is the LayerNorm input M_k + A_k.
    for i, layer in enumerate(model.encoder.layer):
        hooks.append(layer.output.dense.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))    
    
    hidden_size = model.config.hidden_size

    a1 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    if accurate:
        print("ACCURATE MODE IS ON (MORE MEMORY IS NEEDED")
        a3 = torch.empty(
            (dataset_size * max_length, model.config.hidden_size),
            dtype=torch.bfloat16,
            device='cpu'
        )
    
    cnt = 0
    for batch in tqdm(
        dataloader,
        desc=Fore.RED + "Gathering Activations" + Fore.RESET,
        dynamic_ncols=True,
        colour="red"
    ):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        inputs = {k: batch[k] for k in ("input_ids", "attention_mask")}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # After [1:], hidden_states[k] is the output of 0-based layer k. With idx_i = start_id - num_layer - 1
        # (the last layer that is KEPT) and idx_n = end_id - num_layer - 1 (the last layer that is REMOVED),
        # the block idx_i+1 .. idx_n is cut, and the goal is: new output of layer idx_i ≈ h_n.
        # num_layer shifts the ids when earlier blocks were already cut from model_path.
        hidden_states = outputs.hidden_states[1:]
        hidden_states_mlp_list = [
            mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        ]
        hidden_states_mlp = hidden_states_mlp_list[start_id - num_layer - 1]

        hidden_states_i = hidden_states[start_id - num_layer - 1]
        hidden_states_n = hidden_states[end_id - num_layer - 1]

        # Reshape activations
        hidden_states_mlp = hidden_states_mlp.reshape(-1, hidden_size).to(torch.float64)
        hidden_states_i = hidden_states_i.reshape(-1, hidden_size).to(torch.float64)
        hidden_states_n = hidden_states_n.reshape(-1, hidden_size).to(torch.float64)
        
        # Residual math inherited from ReplaceMe's Llama (pre-norm) code. There, h_i = r_i + M_i, where r_i
        # is the residual stream after attention, so h_i - M_i recovers r_i exactly. The fit then targets
        #   non-accurate: cosine(M_i T, h_n - r_i)    -> a1 = M_i, a2 = h_n + M_i - h_i
        #   accurate:     cosine(M_i T + r_i, h_n)    -> a1 = M_i, a2 = h_n,  a3 = h_i - M_i
        # DeBERTa is post-LN: h_i = LN_i(M_i + A_i). So h_i - M_i = LN_i(M_i + A_i) - M_i, which is NOT the
        # residual A_i: it mixes a normalized vector with an unnormalized one. Both a2 (non-accurate) and a3
        # (accurate) are therefore built from this wrong residual. Also, after the fold below the new layer
        # output is LN_i(M_i T + A_i): the LayerNorm sits after the sum, yet the loss compares
        # M_i T (+ a3) directly with h_n, which is an LN output. The loss does not model the actual
        # post-pruning forward pass.
        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n + hidden_states_mlp - hidden_states_i
        if accurate:
            a2_batch = hidden_states_n
            a3_batch = hidden_states_i - hidden_states_mlp
            a3[cnt:cnt+a3_batch.shape[0]] = a3_batch
        a1[cnt:cnt+a1_batch.shape[0]] = a1_batch
        a2[cnt:cnt+a2_batch.shape[0]] = a2_batch
        
        cnt += a2_batch.shape[0]
        
        del hidden_states_mlp, hidden_states_i, hidden_states_n
    
    a1 = a1[:cnt]
    a2 = a2[:cnt]
    if accurate:
        a3 = a3[:cnt]    
    if solver == "adam":
        transform = adam_method(a1, a2, a3=a3 if accurate else None, loss=loss, diag=diag, two_vectors=two_vectors, thri=thri)
    else:
        transform = optimizing_method(a1, a2, a3=a3 if accurate else None, solver=solver)
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

     # Load fresh GLiNER2 model for transformation (kept in FP32: GLiNER2 heads expect FP32 encoder outputs)
    del extractor
    extractor = AutoExtractor.from_pretrained(model_path)
    extractor.to("cpu")
    extractor.eval()
    model = extractor.encoder
    
    # Removes 0-based layers [idx_i + 1, idx_n + 1); layer idx_i is kept and now feeds layer idx_n + 1 directly
    model = truncate_model(model, start_id - num_layer, end_id - num_layer)
    
    # Save results
    if save_path is None:
        os.makedirs('output_models', exist_ok=True)
        save_path = os.path.join(
            "output_models",
            f"{model_path}_{layers_to_skip}_layers_{start_id}_{end_id}_{dataset}_{dataset_size}".replace("/", "_")
        )
    
    # Apply transformation: fold T into layer idx_i's FFN down-projection, so that dense'(x) = dense(x) T.
    # Layer idx_i then outputs LN_i(M_i T + A_i). LN_i keeps its own gamma/beta, while the target h_n was
    # produced by LN_n's gamma/beta.
    dense = extractor.encoder.encoder.layer[start_id - num_layer - 1].output.dense
    transform_t = transform.T.cpu()
    dense.load_state_dict({
        "weight": (transform_t @ dense.weight.to(torch.float64)).to(dense.weight.dtype),
        "bias": (transform_t @ dense.bias.to(torch.float64)).to(dense.bias.dtype)
    })

    # Writes config.json, encoder_config/ (with the reduced num_hidden_layers), model.safetensors and the tokenizer
    extractor.save_pretrained(f"{save_path}_ReplaceMe_{loss}_{solver}")

    if save_transform_only:
        torch.save(transform, f"{save_path}_ReplaceMe_{loss}_{solver}_transform")
        
    # Final cleanup
    del model, a1, a2
    if accurate:
      del a3
    gc.collect()
    torch.cuda.empty_cache()
    
    return f"{save_path}_ReplaceMe_{loss}_{solver}"


def read_config(config_path: str) -> dict:
    """Read and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_from_config():
    """Run the cosine distance calculation from a configuration file."""
    parser = argparse.ArgumentParser(
        description="Run numerical solvers for linear transform estimation based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    config = read_config(args.config)
    seed_all(config.pop("seed", 42))

    average_distances = torch.load(config['distances_path'])
    selected_blocks = select_non_overlapping_blocks(
        average_distances,
        config['layers_to_skip'],
        num_blocks=config['num_A'],
        merge_consecutive=config['merge_consecutive']
    )
    
    start_ids = sorted([x[0] for x in selected_blocks])
    end_ids = sorted([x[1] for x in selected_blocks])
    num_layers = [end_ids[i] - start_ids[i] for i in range(len(start_ids))]
    num_layers = [sum(num_layers[:i]) for i in range(len(start_ids)+1)]
    
    for i in range(len(selected_blocks)):
        path = cosine_dist(**config, start_id=start_ids[i], end_id=end_ids[i], num_layer=num_layers[i], selected_blocks=selected_blocks)
        config["model_path"] = path