import argparse
import os
import sys
import json
import math
import random
import numpy as np
import torch
from itertools import chain
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    CONFIG_MAPPING,
    MODEL_MAPPING,
)
from transformers.utils import (
    check_min_version,
    get_full_repo_name,
    send_example_telemetry,
)
from transformers.utils.versions import require_version

from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from lib.prune_all import prune_fixed, prune_flow, check_sparsity, find_layers
from lib.eval import eval_ppl

# Logging setup
logger = get_logger(__name__)

# Ensure required version of datasets library
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

# Print GPU information
print('# of GPUs: ', torch.cuda.device_count())

# Model configuration constants
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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

def main():
    ########################## FLOW Pruning ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=256, help='Number of calibration samples.')
    parser.add_argument("--sparsity_type", default="4:8", type=str)
    parser.add_argument("--prune_method", type=str, help="Options: 'from_file', 'fixed'")
    parser.add_argument('--input_file', type=str, default = "", help='Path to the input file.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="The configuration name of the dataset to use (via the datasets library).") 
    parser.add_argument("--Lamda", default=0.2, type=float, help="Lamda")
    parser.add_argument("--Hyper_m", type=float, default=3)
    parser.add_argument("--outlier_by_activation", action="store_true", help="outlier_by_activation")
    parser.add_argument("--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")
  
   
    args = parser.parse_args()

    print(f"Number of calibration samples (nsamples): {args.nsamples}")
    # Setting seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity for fixed pruning
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.prune_method == "fixed":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"Loading LLM model: {args.model}")
    model = get_llm(args.model, args.cache_dir)
    
    print("Model loaded successfully!")
    print("=" * 80)
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Model Details:\n{model}")
    
    model.eval()
    
    if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    elif "llama" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)



    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]

    print(f"Using device: {device}")
    if args.prune_method != "fixed":
        print("Pruning model using method:", args.prune_method, args.input_file)
    else:
        print("Pruning model using method:", args.prune_method, f"{prune_n}:{prune_m}")


    ############################ baseline   ############################
    if args.prune_method == "from_file":
        model = prune_flow(args, model, tokenizer, device)
    elif args.prune_method == "fixed":
        model = prune_fixed(args, model, tokenizer, device, N=prune_n, M=prune_m)
    else:
        model = model


    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")

    sys.stdout.flush()
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")

if __name__ == '__main__':
    main()
