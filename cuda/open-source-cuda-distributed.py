import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse


def setup_distributed():
    """Set up PyTorch distributed environment."""
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    node_list = os.environ["SLURM_NODELIST"]
    master_addr = (
        os.popen(f"scontrol show hostname {node_list} | head -n1").read().strip()
    )

    # Derive a unique port for this job to avoid collisions
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    base_port = 29500  # Base port number
    port_offset = int(job_id) % 1000  # Ensure offset is within a reasonable range
    master_port = str(base_port + port_offset)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl")


def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model = DDP(model, device_ids=[device], output_device=device)
    return model, tokenizer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    args = parser.parse_args()

    setup_distributed()

    # Setup device for the current process
    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    pipe = pipeline(
        "text-generation", model=model.module, tokenizer=tokenizer, device=device
    )

    if torch.distributed.get_rank() == 0:
        # Only the first process handles the prompt to avoid duplicate outputs
        prompts = [
            "As a data scientist, can you explain the concept of regularization in machine learning?"
        ]
    else:
        prompts = []

    # Distribute prompts (for demonstration, assuming distribution is handled externally)
    for prompt in prompts:
        sequences = pipe(prompt, max_length=args.num_tokens, num_return_sequences=1)
        print(sequences[0]["generated_text"])


if __name__ == "__main__":
    main()
