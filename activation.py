from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
import subprocess
import threading
import re
import pandas as pd
from time import sleep
import time
import argparse
import os
import datetime
import torch.mps
import numpy as np
from scipy import stats
import torch


model_name = "mistralai/Mistral-7B-v0.1"  # Example using GPT-2, but you can choose any suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode


def count_active_parameters(model, input_tensor):
    # Use a dictionary to store the number of activated parameters
    activated_params = {"count": 0}

    def forward_hook(module, inp, out):
        if hasattr(module, "weight"):
            # Add the number of elements in the weight tensor to the count
            activated_params["count"] += module.weight.numel()

    # Register hook for all modules
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):  # Only add hooks to modules with weights
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    # Run the model to trigger hooks
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks after use to clean up
    for hook in hooks:
        hook.remove()

    # Return the total count of activated parameters
    return activated_params["count"]


input_text = "Hello, my name is"
input_tensor = tokenizer(input_text, return_tensors="pt")["input_ids"]

active = count_active_parameters(model, input_tensor)

print(f"Number of active parameters: {active}")
