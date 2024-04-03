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

df = df[df["Phase"].str.contains("inference")]  # Filter for only inference jobs
df = df[df["Phase"].str.contains("0")]  # Remove the first inference job for consistency
df_only_mac = df[df["System"] == "M1-Pro"]
# df = df[df["System"] != "M1-Pro"]

sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_palette("colorblind")
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    data=df_only_mac,
)
plt.savefig("input-tokens-throughput-mac.pdf", bbox_inches="tight")

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Number of Input Tokens",
    y="Energy per Token (J/tokens)",
    data=df_only_mac,
)
plt.savefig("input-tokens-energy-per-token-mac.pdf", bbox_inches="tight")

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

plt.figure(figsize=(10, 6))
facet_grid = sns.catplot(
    x="Number of Input Tokens",
    y="Throughput (tokens/s)",
    hue="Model",
    col="System",
    data=df,
    kind="point",
    sharey=False,
    col_order=["M1-Pro", "Palmetto Intel+V100", "Swing AMD+A100"],
)
facet_grid.set_xticklabels(rotation=45)
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
