from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.handler.pandas_handler import PandasHandler
import argparse
import datetime
import pandas as pd
from pynvml.smi import nvidia_smi


def tokenizer_model_pipeline(
    model_name: str,
    ctx: EnergyContext,
) -> tuple[Pipeline, AutoTokenizer]:

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
    return pipe, tokenizer


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
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
    return sequences[0]["generated_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=256)
    parser.add_argument("--hf_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--system_name", type=str, default="Swing")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    num_gpus = torch.cuda.device_count()
    hf_name = args.hf_name
    model_name = hf_name.split("/")[1]
    num_tokens = args.num_tokens
    batch_size = args.batch_size
    pandas_handle = PandasHandler()
    # profile_tokenizer = ProfileAMDEnergy(
    #     tag="tokenizer-model-pipeline",
    #     date=todays_date,
    #     model=model_name,
    #     system_name=args.system_name,
    #     num_gpus=num_gpus,
    #     num_tokens=num_tokens,
    #     batch_size=batch_size,
    # )
    # profile_tokenizer_proc = profile_tokenizer.start_profiling()
    with EnergyContext(
        handler=pandas_handle,
        domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
        start_tag="tokenizer",
    ) as ctx:
        pipe, tokenizer = tokenizer_model_pipeline(args.hf_name, ctx)
    # profile_tokenizer.stop_profiling(proc=profile_tokenizer_proc)
    df = pandas_handle.get_dataframe()
    df["Max Number of Tokens"] = num_tokens
    df["Input Tokens"] = 0
    df["Iteration"] = 0
    df["Model Name"] = model_name
    df["Number of GPUs"] = num_gpus
    df["Prompt"] = "startup"
    df["Output Tokens"] = 0
    df["Batch Size"] = batch_size
    df["System Name"] = args.system_name
    for idx in range(num_gpus):
        df[f"Total Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
            "memory.total"
        )["gpu"][idx]["fb_memory_usage"]["total"]
        df[f"Used Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery("memory.used")[
            "gpu"
        ][idx]["fb_memory_usage"]["used"]
    df.to_csv(
        f"{model_name}-{args.system_name}-{num_gpus}.csv",
        mode="a",
        header=False,
        index=False,
    )
    prompts = {
        "A": "What is the capital of France?",
        "B": "Can you explain the difference between a simile and a metaphor? Provide an example of each.",
        "C": "What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic or a personal crisis?",
        "D": "Imagine you are a travel guide. Can you recommend a 7-day itinerary for a trip to Japan, including must-visit destinations, cultural experiences, and local cuisine? Provide a brief description of each day's activities and how they showcase the best of Japan.",
        "E": "As an AI language model, you have the ability to process and generate human-like text. Can you discuss the potential implications of advanced AI systems like yourself on various industries, such as healthcare, education, and creative fields? Consider the benefits, challenges, and ethical considerations surrounding the integration of AI in these sectors. Provide specific examples to support your analysis.",
    }

    for idx, prompt in prompts.items():
        max_iterations = 5
        iteration = 0
        previous_var = float("inf")

        while iteration < max_iterations:
            pandas_handle = PandasHandler()
            idx_log = (idx, iteration)
            # profile_inference = ProfileAMDEnergy(
            #     tag=f"inference-{idx_log[0]}-{idx_log[1]}",
            #     date=todays_date,
            #     model=model_name,
            #     system_name=args.system_name,
            #     num_gpus=num_gpus,
            #     num_tokens=num_tokens,
            #     batch_size=batch_size,
            # )
            # profile_inference_proc = profile_inference.start_profiling()
            with EnergyContext(
                handler=pandas_handle,
                domains=[NvidiaGPUDomain(i) for i in range(num_gpus)],
                start_tag=f"start-inference-{idx_log[0]}-{idx_log[1]}",
            ) as ctx:
                llm_output = run_inference(pipe, num_tokens, prompt)
            print(llm_output)
            # profile_inference.stop_profiling(proc=profile_inference_proc)
            input_tokens = tokenizer.encode(prompt)
            num_input_tokens = len(input_tokens)
            output_tokens = tokenizer.encode(llm_output)
            num_output_tokens = len(output_tokens)
            df = pandas_handle.get_dataframe()
            df["Max Number of Tokens"] = num_tokens
            df["Input Tokens"] = num_input_tokens
            df["Iteration"] = iteration
            df["Model Name"] = model_name
            df["Number of GPUs"] = num_gpus
            df["Prompt"] = prompt
            df["Output Tokens"] = num_output_tokens
            df["Batch Size"] = batch_size
            df["System Name"] = args.system_name
            for idx in range(num_gpus):
                df[f"Total Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.total"
                )["gpu"][idx]["fb_memory_usage"]["total"]
                df[f"Used Memory {idx}"] = nvidia_smi.getInstance().DeviceQuery(
                    "memory.used"
                )["gpu"][idx]["fb_memory_usage"]["used"]

            df.to_csv(
                f"{model_name}-{args.system_name}-{num_gpus}.csv",
                mode="a",
                header=False,
                index=False,
            )
            current_var = df["nvidia_gpu_0"].std()
            if abs(previous_var - current_var) < 0.1:
                break
            previous_var = current_var
            iteration += 1
