import sys
sys.path.append("../")
import time 
import heapq 
import torch 
import torch.nn as nn 
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders 
import numpy as np
from pdb import set_trace as st
import argparse
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
from collections import defaultdict

def check_sparsity(N,M):
    """
    Calculate the sum of N/M ratios and average across all layers.

    Args:
        N (list): List of N values for each layer.
        M (list): List of M values for each layer.

    Returns:
        float: The average of N/M ratios across all layers.
    """
    if len(N) != len(M):
        raise ValueError("N and M must have the same length.")

    nm_ratios = [(m-n) / m for n, m in zip(N, M) if m != 0]
    sparsity_ratio = sum(nm_ratios) / len(nm_ratios) if nm_ratios else 0.0

    return sparsity_ratio

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def normalize_to_range(data, lower=0, upper=1):
    """
    Normalize a list of numerical values to a specified range [lower, upper].

    Parameters:
        data (list of float): The list of numerical values to be normalized.
        lower (float, optional): The lower bound of the target range. Default is 0.
        upper (float, optional): The upper bound of the target range. Default is 1.

    Returns:
        list of float: A list of normalized values scaled to the specified range.
    """
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) * (upper - lower) + lower for x in data]

def check_outlier_mean(mask,threshold):
    """
    Calculates the outlier ratio of elements in a tensor based on a threshold 
    relative to the mean of the tensor.
    Args:
        mask (torch.Tensor): The input tensor to analyze.
        threshold (float): The multiplier for the mean to determine the outlier threshold.
    Returns:
        float: The percentage of elements in the tensor that are considered outliers.
    """
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio

def calculate_outlier_locality_index(mask, threshold, subset_size=10000):
    """
    Calculates the outlier locality index for a given mask tensor. The index is a measure of how 
    clustered or spread out the outliers are, based on their pairwise distances.
    Args:
        mask (torch.Tensor): A tensor representing the data mask, where outliers are identified.
        threshold (float): A multiplier for the mean of the mask to determine the outlier threshold.
        subset_size (int, optional): The maximum number of outliers to consider for the calculation. 
                                     Defaults to 10000.
    Returns:
        float: The outlier locality index. Higher values indicate more clustered outliers, while 
               lower values indicate more spread-out outliers. Returns 0.0 if there are fewer than 
               two outliers.
    """
    W = mask
    # Identify outliers based on the threshold
    max_threshold = torch.mean(W) * threshold
    outlier_positions = (W > max_threshold).nonzero(as_tuple=False)

    num_outliers = len(outlier_positions)
    
    if num_outliers < 2:
        # If there are fewer than 2 outliers, we can't calculate distances, so return 0 or some other indicator
        return 0.0

    # Take a contiguous subset of outliers
    if num_outliers > subset_size:
        sampled_outlier_positions = outlier_positions[:subset_size]
    else:
        sampled_outlier_positions = outlier_positions

    num_sampled_outliers = len(sampled_outlier_positions)

    if num_sampled_outliers < 2:
        return 0.0

    # Compute pairwise distances between sampled outliers (Euclidean distance)
    distances = torch.cdist(sampled_outlier_positions.float(), sampled_outlier_positions.float(), p=2)
    
    # Get the upper triangular part of the distance matrix, excluding the diagonal (self-distances)
    distance_sum = distances.triu(diagonal=1).sum().item()
    
    # Calculate the average distance between sampled outliers
    avg_distance = distance_sum / (num_sampled_outliers * (num_sampled_outliers - 1) / 2)

    # To convert this into a locality index, we can invert the average distance
    # Higher values indicate more clustered outliers, lower values indicate more spread out
    outlier_locality_index = 1.0 / avg_distance    
    return outlier_locality_index

def ilp_solver(O,D, S_target):
    """
    Solves FLOW Integer Linear Programming (ILP) problem to optimize layer-wise sparsity 
    while adhering to power-of-two constraints for N and M combinations.

    Args:
        O (list): OWL Like metric.
        D (list): FLOW Outlier Locality Index.
        S_target (float): The target average sparsity across all layers. 
                            Should be a value in the range [0, 1].

    Returns:
        tuple: A tuple containing two lists:
            - N (list of int): The selected power-of-two values for N for each layer.
            - M (list of int): The selected power-of-two values for M for each layer.
    """
    # Check if normalization is needed
    print("Length", len(O))
    l = len(O)
    if any(x < 0 or x > 1 for x in O + D):
        O = normalize_to_range(O)
        D = normalize_to_range(D)

    # Problem definition
    problem = LpProblem("Layerwise_Sparsity_Optimization", LpMinimize)

    # Precompute valid sparsity values for N and M combinations
    powers_of_two_N = [4, 2, 8]
    powers_of_two_M = [8, 4]
    sparsity_values = {
        (n, m): (m - n) / m for n in powers_of_two_N for m in powers_of_two_M if n <= m
    }

    # Variables
    # Binary variables to enforce power-of-two constraints
    z = {
        (i, n, m): LpVariable(f"z_{i}_{n}_{m}", cat=LpBinary)
        for i in range(l)
        for n in powers_of_two_N
        for m in powers_of_two_M
        if (n, m) in sparsity_values
    }

    # Decision variables for layer sparsity
    S = [LpVariable(f"S_{i}", lowBound=0, upBound=1) for i in range(l)]

    # Objective function
    alpha, beta = 1, 4  # Hyperparameters
    objective = lpSum(
        lpSum(
            z[(i, n, m)]
            * (alpha * abs(n - O[i] * max(powers_of_two_N)) + beta * abs(m - (1 - D[i]) * max(powers_of_two_M)))
            for (n, m) in sparsity_values
        )
        for i in range(l)
    )
    problem += objective

    # Constraints
    # Ensure one combination of (N, M) is chosen per layer
    for i in range(l):
        problem += lpSum(z[(i, n, m)] for (n, m) in sparsity_values if (i, n, m) in z) == 1

    # Target sparsity
    problem += (
        lpSum(
            lpSum(z[(i, n, m)] * sparsity_values[(n, m)] for (n, m) in sparsity_values if (i, n, m) in z)
            for i in range(l)
        )
        / l
        == S_target
    )

    # Layer sparsity relationship
    for i in range(l):
        problem += S[i] == lpSum(
            z[(i, n, m)] * sparsity_values[(n, m)] for (n, m) in sparsity_values if (i, n, m) in z
        )

    # Solve
    problem.solve()

    N = []
    M = []
    # Output results
    for i in range(l):
        selected_n, selected_m = None, None
        for (n, m) in sparsity_values:
            if (i, n, m) in z and z[(i, n, m)].varValue == 1:
                selected_n, selected_m = n, m
                N.append(selected_n)
                M.append(selected_m)

    # Pad with default values if solver failed to find complete assignments
    while len(N) < l:
        N.append(4)
        M.append(8)
    
    return N,M

def prepare_calibration_input_opt(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids

def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def solver_main(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Main function to solve the ILP problem.

    Args:
        args: Command-line arguments.
        model: The model to select pruning ratios.
        tokenizer: The tokenizer for the model.
        device: The device to run the model on.
        prune_n: The number of layers to prune.
        prune_m: The number of heads to prune.

    Returns:
        tuple: A tuple containing two lists:
            - N (list of int): The selected power-of-two values for N for each layer.
            - M (list of int): The selected power-of-two values for M for each layer.
    """
    all_layer_ratio=[]
    all_layer_outlier_locality_index=[] 
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(args, model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"Identifying metric for layer {i}, name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data) # Will only be used if pruning is done by outlier activation
            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            outlier_locality_index_layer = calculate_outlier_locality_index(layer_wmetric, out_ratio)

        all_layer_ratio.append(out_ratio_layer)
        all_layer_outlier_locality_index.append(outlier_locality_index_layer)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    # Adjust range of all_layer_outlier_locality_index and all_layer_ratio
    all_layer_outlier_locality_index = np.array(all_layer_outlier_locality_index)
    all_layer_outlier_locality_index = args.Lamda*(all_layer_outlier_locality_index - all_layer_outlier_locality_index.min()) / (all_layer_outlier_locality_index.max() - all_layer_outlier_locality_index.min())
    all_layer_outlier_locality_index=all_layer_outlier_locality_index-np.mean(all_layer_outlier_locality_index)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)

    N, M = ilp_solver(all_layer_ratio, all_layer_outlier_locality_index, args.sparsity_ratio)

    return N, M


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILP Solver")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Model name")
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=256, help="Number of samples for calibration")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="The name of the dataset to use (via the datasets library)."), 
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5, help="Target global sparsity ratio")
    parser.add_argument("--Hyper_m", type=float, default=0.5, help="Hyperparameter for outlier detection")
    parser.add_argument("--Lamda", type=float, default=0.2, help="Lambda parameter for scaling")
    parser.add_argument("--output_file", type=str, default="layerwise_sparsity.txt", help="Name of the output file to save layerwise N:M ratios")
    args = parser.parse_args()

    # Setting seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"Loading LLM model: {args.model}")
    model = get_llm(args.model, args.cache_dir)
    
    print("Model loaded successfully!")
    print("=" * 80)
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Model Details:\n{model}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    elif "llama" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:  # For 30B and 65B models, use device_map to load onto multiple A6000 GPUs.
        device = model.hf_device_map["lm_head"]

    print(f"Using device: {device}")
    print(f"Target sparsity ratio: {args.sparsity_ratio}")
    print("Starting layer-wise N:M selection process...")

    N,M = solver_main(args, model, tokenizer, device=device)
    print("=" * 80)
    print("Layer-wise N:M selection completed!")

    # Checking if it closely matches the target sparsity ratio
    print("*"*30)
    sparsity_ratio = check_sparsity(N,M)
    print(f"sparsity check {sparsity_ratio:.4f}")
    print("*"*30)

    # Save the selected N:M ratios to a file
    with open(args.output_file, "w") as f:
        for i in range(len(N)):
            f.write(f"Layer {i}: {N[i]}:{M[i]}\n")
    print(f"Layer-wise N:M ratios saved to {args.output_file}")
    print("Exiting!")
    print("=" * 80)