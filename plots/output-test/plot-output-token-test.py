import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Load the data
# df_palmetto = pd.read_csv("input-token-test-palmetto.csv")
# df_swing = pd.read_csv("input-token-test-swing.csv")
# df_mac = pd.read_csv("input-token-test-mac.csv")

# df = pd.concat([df_palmetto, df_swing])

# df["GPU Energy (J)"] = df["GPU Energy (uJ)"] / 1e6

# df = pd.concat([df, df_mac])

# print(df)

sns.set(style="whitegrid", context="talk", font_scale=1.5)
sns.set_palette("colorblind")
plt.figure(figsize=(10, 6))

df = pd.read_csv("all-output-stats.csv")

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
df = df[
    ~df["Phase"].str.contains("-0")
]  # Remove the first inference job for consistency
df = df[df["System"] != "Palmetto Intel+A100"]


plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Output Tokens",
    y="Energy per Token (J/tokens)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    sharex=False,
    kind="bar",
    col_order=[
        "M1-Pro",
        "Palmetto Intel+V100",
        "Swing AMD+A100",
    ],
)
facet_grid.set_xticklabels(rotation=45)
facet_grid.set(yscale="log")
for ax in facet_grid.axes.flat:
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.5)

plt.savefig("output-tokens-energy-per-token.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Output Tokens",
    y="Energy per Token (J/tokens)",
    style="Model",
    hue="System",
    data=df,
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=12,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    # legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(1e-1, 1e3)
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
    alignment="left",
    ncol=2,
)
plt.savefig("output-tokens-energy-per-token-line.pdf", bbox_inches="tight")


plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Output Tokens",
    y="Throughput (tokens/s)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    sharex=False,
    kind="bar",
    col_order=[
        "M1-Pro",
        "Palmetto Intel+V100",
        "Swing AMD+A100",
    ],
)
facet_grid.set_xticklabels(rotation=45)
for ax in facet_grid.axes.flat:
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.5)
plt.savefig("output-tokens-throughput-bar.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Output Tokens",
    y="Throughput (tokens/s)",
    style="Model",
    hue="System",
    data=df,
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=12,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(1e-1, 1e3)
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")


plt.figure(figsize=(10, 5))
facet_grid = sns.catplot(
    x="Number of Output Tokens",
    y="Total Energy (J)",
    hue="Model",
    col="System",
    data=df,
    kind="bar",
    sharey=False,
    sharex=False,
    col_order=[
        "M1-Pro",
        "Palmetto Intel+V100",
        "Swing AMD+A100",
    ],
)
facet_grid.set_xticklabels(rotation=45)
facet_grid.set(yscale="log")
for ax in facet_grid.axes.flat:
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.5)
plt.savefig("output-tokens-total-energy.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Output Tokens",
    y="Total Energy (J)",
    style="Model",
    hue="System",
    data=df,
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=12,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    # legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(1e1, 1e6)
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
    alignment="left",
    ncol=2,
)
plt.savefig("output-tokens-total-energy-line.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Output Tokens",
    y="Average Total Power Draw (W)",
    hue="Model",
    col="System",
    data=df,
    sharey=False,
    sharex=False,
    kind="bar",
    col_order=[
        "M1-Pro",
        "Palmetto Intel+V100",
        "Swing AMD+A100",
    ],
)
facet_grid.set_xticklabels(rotation=45)
for ax in facet_grid.axes.flat:
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)
plt.savefig("output-tokens-total-power-draw.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Output Tokens",
    y="Runtime (s)",
    hue="Model",
    col="System",
    data=df,
    kind="bar",
    sharey=False,
    sharex=False,
    col_order=[
        "M1-Pro",
        "Palmetto Intel+V100",
        "Swing AMD+A100",
    ],
)

facet_grid.set(yscale="log")
for ax in facet_grid.axes.flat:
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.5)

facet_grid.set_xticklabels(rotation=45)
plt.savefig("output-tokens-runtime.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Output Tokens",
    y="Runtime (s)",
    style="Model",
    hue="System",
    data=df,
    hue_order=[
        "Swing AMD+A100",
        "Palmetto Intel+V100",
        "M1-Pro",
    ],
    marker="o",
    markersize=12,
    style_order=["Falcon (7B)", "Llama-2 (7B)", "Mistral (7B)"],
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(1e-1, 1e4)
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("output-tokens-runtime-line.pdf", bbox_inches="tight")
