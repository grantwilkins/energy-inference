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

df = pd.read_csv("all-input-scale-stats.csv")


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
    ~df["Phase"].str.contains("0")
]  # Remove the first inference job for consistency
# df = df[df["System"] != "M1-Pro"]

sns.set(style="whitegrid", context="talk", font_scale=1.5)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Energy per Token (J/tokens)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
# plt.ylim(1e0, 1e2)
# plt.xticks(
#     [
#         8,
#         16,
#         32,
#         64,
#         128,
#         256,
#         512,
#         1024,
#         2048,
#     ]
# )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-scale-tokens-energy-per-token-line.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Energy per Token (J/tokens)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(9e-1, 1e2)
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
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
    alignment="left",
    ncol=2,
)
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
plt.savefig("input-scale-tokens-energy-per-token-log.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Total Energy (J)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
# plt.ylim(1e0, 1e2)
# plt.xticks(
#     [
#         8,
#         16,
#         32,
#         64,
#         128,
#         256,
#         512,
#         1024,
#         2048,
#     ]
# )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-scale-tokens-total-energy-line.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Total Energy (J)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
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
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    frameon=False,
    alignment="left",
    ncol=2,
)
plt.savefig("input-scale-tokens-total-energy-log.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Average Total Power Draw (W)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
# plt.ylim(1e0, 1e2)
# plt.xticks(
#     [
#         8,
#         16,
#         32,
#         64,
#         128,
#         256,
#         512,
#         1024,
#         2048,
#     ]
# )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-scale-tokens-total-power-draw-line.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Runtime (s)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
# plt.ylim(1e0, 1e2)
# plt.xticks(
#     [
#         8,
#         16,
#         32,
#         64,
#         128,
#         256,
#         512,
#         1024,
#         2048,
#     ]
# )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-scale-tokens-runtime-line.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Runtime (s)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(7e-1, 1e2)
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

plt.savefig("input-scale-tokens-runtime-log.pdf", bbox_inches="tight")


sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    # legend=False,
)
# plt.xscale("log", base=2)
# plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
# plt.ylim(1e0, 1e2)
# plt.xticks(
#     [
#         8,
#         16,
#         32,
#         64,
#         128,
#         256,
#         512,
#         1024,
#         2048,
#     ]
# )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig("output-tokens-throughput-line.pdf", bbox_inches="tight")
plt.savefig("input-scale-tokens-throughput-line.pdf", bbox_inches="tight")

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")

plt.figure(figsize=(10, 6))
sns.lineplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    hue="Model",
    marker="o",
    markersize=12,
    data=df,
    hue_order=[
        "Falcon (7B)",
        "Falcon (40B)",
        "Llama-2 (7B)",
        "Llama-2 (13B)",
        "Llama-2 (70B)",
        "Mistral (7B)",
        "Mixtral (8x7B)",
    ],
    legend=False,
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.4, alpha=0.25)
plt.ylim(5e0, 1e3)
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
# plt.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, -0.15),
#     frameon=False,
#     alignment="left",
#     ncol=2,
# )
plt.savefig("input-scale-tokens-throughput-log.pdf", bbox_inches="tight")