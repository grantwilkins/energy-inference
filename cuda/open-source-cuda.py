from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
import argparse
import datetime
import pandas as pd


def tokenizer_model_pipeline(model_name: str, ctx: EnergyContext) -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ctx.record(tag="model load")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ctx.record(tag="pipeline load")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
    idx: tuple[str, int],
    ctx: EnergyContext,
) -> str:
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    ctx.record(tag=f"stop-inference-{idx[0]}-{idx[1]}")
    return sequences[0]["generated_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=3500)
    parser.add_argument("--hf_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--system_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    num_gpus = torch.cuda.device_count()
    hf_name = args.hf_name
    model_name = hf_name.split("/")[1]
    num_tokens = args.num_tokens
    batch_size = args.batch_size

    pandas_handle = PandasHandler()
    with EnergyContext(
        handler=pandas_handle,
        domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
        start_tag="tokenizer",
    ) as ctx:
        pipe = tokenizer_model_pipeline(args.hf_name, ctx)
        ctx.record("startup-done")
    df = pandas_handle.get_dataframe()
    df["Number of Tokens Allowed"] = num_tokens
    df["Length of Input"] = 0
    df["Iteration"] = 0
    df["Model Name"] = model_name
    df["Number of GPUs"] = num_gpus
    df["Prompt"] = "startup"
    df["Number of Tokens Produced"] = 0
    df["Batch Size"] = batch_size
    df.to_csv(
        f"{model_name}-{todays_date}-{num_gpus}.csv",
        mode="a",
        header=False,
        index=False,
    )
    prompts = {
        "A": "Once upon a time",
        "B": "The quick brown fox jumps over the lazy dog",
        "C": "The quick brown fox jumps over the lazy dog",
        "D": "The quick brown fox jumps over the lazy dog",
        "E": "The quick brown fox jumps over the lazy dog",
    }

    for idx, prompt in prompts.items():
        max_iterations = 10
        iteration = 0
        previous_var = float("inf")

        while iteration < max_iterations:
            pandas_handle = PandasHandler()
            idx_log = (idx, iteration)
            with EnergyContext(
                handler=pandas_handle,
                domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
                start_tag=f"start-inference-{idx_log[0]}-{idx_log[1]}",
            ) as ctx:
                llm_output = run_inference(pipe, num_tokens, prompt, idx_log, ctx)
            df = pandas_handle.get_dataframe()
            df["Number of Input Tokens Allowed"] = num_tokens
            df["Length of Input"] = len(prompt)
            df["Iteration"] = 0
            df["Model Name"] = model_name
            df["Number of GPUs"] = num_gpus
            df["Prompt"] = prompt
            df["Number of Tokens Produced"] = len(llm_output)
            df["Batch Size"] = batch_size

            df.to_csv(
                f"{model_name}-{todays_date}-{num_gpus}.csv",
                mode="a",
                header=False,
                index=False,
            )
            current_var = df["nvidia_gpu_0"].std()
            if abs(previous_var - current_var) < 0.1:
                break
            previous_var = current_var
            iteration += 1
