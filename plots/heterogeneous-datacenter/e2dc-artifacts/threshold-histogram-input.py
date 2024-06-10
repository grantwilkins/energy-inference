import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def round_to_nearest_power_of_two(n):
    """
    Rounds a number to the nearest power of two.

    Parameters:
    n (int): The number to be rounded.

    Returns:
    int: The nearest power of two.
    """
    return 2 ** round(math.log(n, 2))


# Load the data
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


thresholds = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

total_energies = []

for threshold in thresholds:
    total_energy = 0
    for length, frequency in lengths_frequency.items():
        if length <= threshold:
            total_energy += (
                mean_energy_per_token_m1_pro.get(round_to_nearest_power_of_two(length))
                * frequency
                * length
            )
        else:
            total_energy += (
                mean_energy_per_token_swing.get(round_to_nearest_power_of_two(length))
                * frequency
                * length
            )
    total_energies.append(total_energy / 3.6e6)

just_m1_pro = 0
just_swing = 0
for length, frequency in lengths_frequency.items():
    just_m1_pro += (
        mean_energy_per_token_m1_pro.get(round_to_nearest_power_of_two(length))
        * frequency
        * length
    )
    just_swing += (
        mean_energy_per_token_swing.get(round_to_nearest_power_of_two(length))
        * frequency
        * length
    )

plt.figure(figsize=(10, 4))
sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.axhline(
    y=just_m1_pro / 3.6e6,
    color="r",
    linestyle="--",
    label="M1-Pro Only",
)
plt.axhline(
    y=just_swing / 3.6e6,
    color="g",
    linestyle="--",
    label="Swing AMD+A100 Only",
)

sns.lineplot(
    x=thresholds,
    y=total_energies,
    marker="o",
    markersize=10,
    color="b",
    label="Hybrid System",
)
plt.legend()
plt.xscale("log", base=2)
plt.xlabel("Threshold")
plt.ylabel("Total Energy (kWh)")
plt.savefig("heterogeneous-threshold-plot.pdf", bbox_inches="tight")


total_number_of_tokens = 0
for length, frequency in lengths_frequency.items():
    total_number_of_tokens += frequency * length

runtimes = []
for threshold in thresholds:
    runtime_for_m1_pro = 0
    runtime_for_swing = 0
    for length, frequency in lengths_frequency.items():
        if length <= threshold:
            runtime_for_m1_pro += (
                frequency
                * length
                * mean_runtime_m1_pro.get(round_to_nearest_power_of_two(length))
            )
        else:
            runtime_for_swing += (
                frequency
                * length
                * mean_runtime_swing.get(round_to_nearest_power_of_two(length))
            )
    print(threshold, runtime_for_m1_pro, runtime_for_swing)
    # Assuming mean_throughput_m1_pro and mean_throughput_swing are constants for simplification
    runtime = runtime_for_swing + runtime_for_m1_pro
    runtimes.append(runtime)

just_m1_pro_runtime = 0
just_swing_runtime = 0
for length, frequency in lengths_frequency.items():
    just_m1_pro_runtime += (
        frequency
        * length
        * mean_runtime_m1_pro.get(round_to_nearest_power_of_two(length))
    )
    just_swing_runtime += (
        frequency
        * length
        * mean_runtime_swing.get(round_to_nearest_power_of_two(length))
    )

# m1_pro_constant_throughput = total_number_of_tokens * 10 / just_m1_pro_runtime
# swing_constant_throughput = total_number_of_tokens * 10 / just_swing_runtime

plt.figure(figsize=(10, 4))
sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.axhline(
    y=just_m1_pro_runtime,
    color="r",
    linestyle="--",
    label="M1-Pro Only",
)
plt.axhline(
    y=just_swing_runtime,
    color="g",
    linestyle="--",
    label="Swing AMD+A100 Only",
)

sns.lineplot(
    x=thresholds,
    y=runtimes,
    marker="o",
    markersize=10,
    color="b",
    label="Hybrid System",
)

plt.legend()
plt.xscale("log", base=2)
plt.xlabel("Threshold")
plt.ylabel("Runtime (s)")
plt.savefig("heterogeneous-threshold-runtime-plot.pdf", bbox_inches="tight")
