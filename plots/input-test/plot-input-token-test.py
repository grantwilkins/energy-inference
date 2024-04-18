import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
# df_palmetto = pd.read_csv("input-token-test-palmetto.csv")
# df_swing = pd.read_csv("input-token-test-swing.csv")
# df_mac = pd.read_csv("input-token-test-mac.csv")

# df = pd.concat([df_palmetto, df_swing])

# df["GPU Energy (J)"] = df["GPU Energy (uJ)"] / 1e6

# df = pd.concat([df, df_mac])

# print(df)

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

df = df[df["System"] != "Palmetto Intel+A100"]
df = df[df["Phase"].str.contains("inference")]  # Filter for only inference jobs
df = df[
    ~df["Phase"].str.contains("0")
]  # Remove the first inference job for consistency
# df = df[df["System"] != "M1-Pro"]

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Energy per Token (J/tokens)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    kind="bar",
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set_xticklabels(rotation=45)
plt.savefig("input-tokens-energy-per-token.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.lineplot(
    x="Number of Input Tokens",
    y="Energy per Token (J/tokens)",
    style="Model",
    hue="System",
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=8,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    data=df,
    # legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.ylim(1e0, 1e2)
plt.xticks(
    [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-tokens-energy-per-token-line.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    kind="bar",
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set_xticklabels(rotation=45)
plt.savefig("input-tokens-throughput-bar.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.lineplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    style="Model",
    hue="System",
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=8,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    data=df,
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.ylim(1e0, 1e3)
plt.xticks(
    [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-tokens-throughput-line.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Total Energy (J)",
    hue="Model",
    col="System",
    data=df,
    kind="bar",
    sharey=False,
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set_xticklabels(rotation=45)
facet_grid.set(yscale="log")
plt.savefig("input-tokens-total-energy.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.lineplot(
    x="Number of Input Tokens",
    y="Total Energy (J)",
    style="Model",
    hue="System",
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=8,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    data=df,
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.ylim(1e2, 1e5)
plt.xticks(
    [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-tokens-total-energy-line.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Average Total Power Draw (W)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    kind="bar",
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set_xticklabels(rotation=45)
plt.savefig("input-tokens-total-power-draw.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.lineplot(
    x="Number of Input Tokens",
    y="Average Total Power Draw (W)",
    style="Model",
    hue="System",
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=8,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    data=df,
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.ylim(1e1, 1e3)
plt.xticks(
    [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-tokens-total-power-draw-line.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Runtime (s)",
    hue="Model",
    col="System",
    data=df,
    kind="bar",
    sharey=False,
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set(yscale="log")
facet_grid.set_xticklabels(rotation=45)
plt.savefig("input-tokens-runtime.pdf", bbox_inches="tight")

plt.figure(figsize=(6, 5))
sns.lineplot(
    x="Number of Input Tokens",
    y="Runtime (s)",
    style="Model",
    hue="System",
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=8,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    data=df,
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.ylim(1e-1, 1e3)
plt.xticks(
    [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-tokens-runtime-line.pdf", bbox_inches="tight")
