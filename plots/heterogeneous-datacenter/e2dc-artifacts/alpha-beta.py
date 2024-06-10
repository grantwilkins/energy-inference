import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
import math
import seaborn as sns


def round_to_nearest_power_of_two(n):
    """
    Rounds a number to the nearest power of two.

    Parameters:
    n (int): The number to be rounded.

    Returns:
    int: The nearest power of two.
    """
    return 2 ** round(math.log(n, 2))


alpha_values = np.linspace(
    0.1, 1.0, 50
)  # Alpha values from low to high priority on energy efficiency

df = pd.read_csv("all-input-stats.csv")
df["Throughput (tokens/s)"] = (
    df["Total Number of Tokens"] / df["Runtime (s)"]
)  # Calculate throughput
df["Total Energy (J)"] = (
    df["GPU Energy (J)"] + df["CPU Energy (J)"]
)  # Calculate total energy

df["Average Total Power Draw (W)"] = df["Total Energy (J)"] / df["Runtime (s)"]

df["Energy per Token (J/tokens)"] = (
    df["Total Energy (J)"] / df["Total Number of Tokens"]
)

df = df[df["Phase"].str.contains("inference")]  # Filter for only inference jobs
df = df[~df["Phase"].str.contains("0")]

df_m1_pro = df[df["System"] == "M1-Pro"]
df_swing = df[df["System"] == "Swing AMD+A100"]

# Create dictionaries for mean Energy per Token for each input token size for M1-Pro and Swing systems
mean_energy_per_token_m1_pro = (
    df_m1_pro.groupby("Number of Input Tokens")["Energy per Token (J/tokens)"]
    .mean()
    .to_dict()
)
mean_energy_per_token_swing = (
    df_swing.groupby("Number of Input Tokens")["Energy per Token (J/tokens)"]
    .mean()
    .to_dict()
)

mean_throughput_m1_pro = (
    df_m1_pro.groupby("Number of Input Tokens")["Throughput (tokens/s)"]
    .mean()
    .to_dict()
)
mean_throughput_swing = (
    df_swing.groupby("Number of Input Tokens")["Throughput (tokens/s)"].mean().to_dict()
)

mean_runtime_m1_pro = (
    df_m1_pro.groupby("Number of Input Tokens")["Runtime (s)"].mean()
).to_dict()
mean_runtime_swing = (
    df_swing.groupby("Number of Input Tokens")["Runtime (s)"].mean()
).to_dict()

dataset = load_dataset("vicgalle/alpaca-gpt4")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
lengths_instructions = [
    len(tokenizer.encode(x)) for x in dataset["train"]["instruction"]
]
lengths_inputs = [len(tokenizer.encode(x)) for x in dataset["train"]["input"]]
lengths = [x + y for x, y in zip(lengths_instructions, lengths_inputs)]
# Generate a frequency dictionary from the lengths
lengths_frequency = {}
for length in lengths:
    if length in lengths_frequency:
        lengths_frequency[length] += 1
    else:
        lengths_frequency[length] = 1

threshold = 32


def utility_function1(
    alpha: float,
    energy: dict,
    runtime: dict,
) -> float:
    beta = 1 - alpha
    total_energy = 0
    total_runtime = 0
    for length, frequency in lengths_frequency.items():
        total_energy += (
            energy.get(round_to_nearest_power_of_two(length)) * frequency * length
        )
        total_runtime += (
            runtime.get(round_to_nearest_power_of_two(length)) * frequency * length
        )
    return alpha * total_energy / (beta * total_runtime)


def utility_function2(
    alpha: float,
    energy_m1: dict,
    energy_swing: dict,
    runtime_m1: dict,
    runtime_swing: dict,
) -> float:
    beta = 1 - alpha
    total_energy_m1 = 0
    total_runtime_m1 = 0
    total_energy_swing = 0
    total_runtime_swing = 0
    for length, frequency in lengths_frequency.items():
        total_energy_m1 += (
            energy_m1.get(round_to_nearest_power_of_two(length)) * frequency * length
        )
        total_runtime_m1 += (
            runtime_m1.get(round_to_nearest_power_of_two(length)) * frequency * length
        )
        total_energy_swing += (
            energy_swing.get(round_to_nearest_power_of_two(length)) * frequency * length
        )
        total_runtime_swing += (
            runtime_swing.get(round_to_nearest_power_of_two(length))
            * frequency
            * length
        )
    return alpha * (total_runtime_m1 + total_energy_m1) + beta * (
        total_runtime_swing * total_energy_swing
    )


utility_vals_m1 = [
    utility_function2(
        alpha,
        mean_energy_per_token_m1_pro,
        mean_energy_per_token_swing,
        mean_runtime_m1_pro,
        mean_runtime_swing,
    )
    for alpha in alpha_values
]
# utility_vals_swing = [
#     utility_function2(
#         alpha,
#         mean_energy_per_token_swing,
#         mean_runtime_swing,
#     )
#     for alpha in alpha_values
# ]

sns.lineplot(x=alpha_values, y=utility_vals_m1)
# sns.lineplot(x=alpha_values, y=utility_vals_swing)
plt.show()
