import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("phi-2-mps.csv")

# Initialize variables
cycle = 0
last_event = None
token_count = 200


# Function to determine test cycles
def define_cycle(row):
    global cycle, last_event
    if last_event == "inference" and row["event"] == "tokenizer":
        cycle += 1
    last_event = row["event"]
    return cycle


# Apply the function to create a cycle column
df["cycle"] = df.apply(define_cycle, axis=1)

# Accumulate timestamps within each cycle
df["accumulated_time"] = df.groupby("cycle")["timestamp"].cumsum() - df.groupby(
    "cycle"
)["timestamp"].transform("first")

# Output the processed DataFrame
print(df)

# Save to new CSV if needed
df.to_csv("processed_data.csv", index=False)
df["CPU PyTorch"] = df["cpu"] * (df["python3.11"] / df["ALL_TASKS"]) * 0.001
df["GPU PyTorch"] = df["gpu"] * 0.001
df["Time (s)"] = df["accumulated_time"] / 1000
max_accumulated_time_per_event = df.groupby("event")["Time (s)"].max()
df["GPU Energy (J)"] = (
    df.groupby("event")
    .apply(lambda x: (x["timestamp"] * x["GPU PyTorch"]).cumsum() / 1000)
    .reset_index(level=0, drop=True)
)
df["CPU Energy (J)"] = (
    df.groupby("event")
    .apply(lambda x: (x["timestamp"] * x["CPU PyTorch"]).cumsum() / 1000)
    .reset_index(level=0, drop=True)
)
df["Model"] = "phi-2"

print(df)

energy_df = df.groupby("event")[["CPU Energy (J)", "GPU Energy (J)"]].max()
energy_df["GPU Energy (J)"] = energy_df["CPU Energy (J)"] + energy_df["GPU Energy (J)"]
sns.set_theme(style="whitegrid")
sns.set_context("paper")
sns.set_color_codes("muted")
plt.figure(figsize=(4, 3))

sns.barplot(
    data=energy_df, x=energy_df.index, y="GPU Energy (J)", color="r", label="GPU"
)
sns.barplot(
    data=energy_df, x=energy_df.index, y="CPU Energy (J)", color="b", label="CPU"
)
plt.legend(loc="best")
plt.xlabel("Event")
plt.ylabel("Energy (J)")
plt.tight_layout()
plt.savefig("phi2-energy-barchart.pdf")


sns.set_theme(style="whitegrid")
sns.set_context("paper")
plt.figure(figsize=(4, 3))
sns.lineplot(
    data=df,
    x="Time (s)",
    y="GPU PyTorch",
    label="GPU PyTorch",
)
sns.lineplot(
    data=df,
    x="Time (s)",
    y="CPU PyTorch",
    label="CPU PyTorch",
)
plt.axvline(x=0, color="red", label="Load the Tokenizer", linestyle="--")
plt.axvline(
    x=max_accumulated_time_per_event["tokenizer"],
    color="red",
    linestyle="--",
)
plt.axvline(
    x=max_accumulated_time_per_event["model load"],
    color="red",
    linestyle="--",
)
plt.axvline(
    x=max_accumulated_time_per_event["pipeline load"],
    color="red",
    linestyle="--",
)
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.savefig("phi-2-mps.pdf")


def aggregate_time(row):
    global time, last_event
    if last_event != row["event"]:
        time = 0
    else:
        time += row["timestamp"]
    last_event = row["event"]
    return time


df["Process Time"] = df.apply(aggregate_time, axis=1) / 1000
df["Token Rate (tokens/s)"] = token_count / df["Process Time"]
df_inference = df[df["event"] == "inference"]
print(df_inference["Token Rate (tokens/s)"].min())
print(df_inference["GPU Energy (J)"].max())
print(df_inference["GPU Energy (J)"].max() / token_count)
