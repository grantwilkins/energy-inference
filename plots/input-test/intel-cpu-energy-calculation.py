import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

core_0_power = 37.5  # W
core_1_power = 37.5  # W

df_falcon_stats = pd.read_csv("falcon-7b-palmetto-a100-1.csv")
df_mistral_stats = pd.read_csv("Mistral-7B-v0.1-palmetto-a100-1.csv")
df_llama_stats = pd.read_csv("Llama-2-7b-chat-hf-palmetto-a100-1.csv")

df_falcon_stats["Idle Package-0 Energy (uJ)"] = (
    core_0_power * df_falcon_stats["Runtime (s)"] * 1e6
)
df_falcon_stats["Idle Package-1 Energy (uJ)"] = (
    core_1_power * df_falcon_stats["Runtime (s)"] * 1e6
)

df_mistral_stats["Idle Package-0 Energy (uJ)"] = (
    core_0_power * df_mistral_stats["Runtime (s)"] * 1e6
)
df_mistral_stats["Idle Package-1 Energy (uJ)"] = (
    core_1_power * df_mistral_stats["Runtime (s)"] * 1e6
)

df_llama_stats["Idle Package-0 Energy (uJ)"] = (
    core_0_power * df_llama_stats["Runtime (s)"] * 1e6
)
df_llama_stats["Idle Package-1 Energy (uJ)"] = (
    core_1_power * df_llama_stats["Runtime (s)"] * 1e6
)

df_falcon_stats["CPU Energy (J)"] = (
    df_falcon_stats["CPU Package-0 Energy (uJ)"]
    + df_falcon_stats["CPU Package-1 Energy (uJ)"]
    - df_falcon_stats["Idle Package-0 Energy (uJ)"]
    - df_falcon_stats["Idle Package-1 Energy (uJ)"]
) / 1e6
df_mistral_stats["CPU Energy (J)"] = (
    df_mistral_stats["CPU Package-0 Energy (uJ)"]
    + df_mistral_stats["CPU Package-1 Energy (uJ)"]
    - df_mistral_stats["Idle Package-0 Energy (uJ)"]
    - df_mistral_stats["Idle Package-1 Energy (uJ)"]
) / 1e6
df_llama_stats["CPU Energy (J)"] = (
    df_llama_stats["CPU Package-0 Energy (uJ)"]
    + df_llama_stats["CPU Package-1 Energy (uJ)"]
    - df_llama_stats["Idle Package-0 Energy (uJ)"]
    - df_llama_stats["Idle Package-1 Energy (uJ)"]
) / 1e6

df_falcon_stats["GPU Energy (J)"] = df_falcon_stats["GPU Energy (uJ)"] / 1e6
df_mistral_stats["GPU Energy (J)"] = df_mistral_stats["GPU Energy (uJ)"] / 1e6
df_llama_stats["GPU Energy (J)"] = df_llama_stats["GPU Energy (uJ)"] / 1e6

print(df_falcon_stats["CPU Energy (J)"])

df_falcon_stats.to_csv(
    "falcon-7b-palmetto-a100-1.csv", sep=",", index=False, header=True
)
df_mistral_stats.to_csv(
    "Mistral-7B-v0.1-palmetto-a100-1.csv", sep=",", index=False, header=True
)
df_llama_stats.to_csv(
    "Llama-2-7b-chat-hf-palmetto-a100-1.csv", sep=",", index=False, header=True
)
