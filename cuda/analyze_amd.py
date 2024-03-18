import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "mistral-7b_power_records.csv"
# Read the data from the file
data = pd.read_csv(file_path, sep=",")
# Extract the core power columns
core_columns = [
    col for col in data.columns if col.startswith("core") and col.endswith("-power")
]

# # Calculate the mean power for each core
# mean_power = data[core_columns].mean()

# # Reshape the mean power values into two 8x8 matrices
# matrix1 = mean_power[:64].values.reshape(8, 8)
# matrix2 = mean_power[64:].values.reshape(8, 8)

# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# # Plot the heatmaps
# heatmap1 = ax1.imshow(matrix1, cmap="hot", interpolation="nearest")
# heatmap2 = ax2.imshow(matrix2, cmap="hot", interpolation="nearest")

# # Add colorbars
# fig.colorbar(heatmap1, ax=ax1)
# fig.colorbar(heatmap2, ax=ax2)

# # Set titles and labels
# ax1.set_title("Mean Energy Usage (Cores 0-63)")
# ax2.set_title("Mean Energy Usage (Cores 64-127)")
# ax1.set_xlabel("Core")
# ax1.set_ylabel("Core")
# ax2.set_xlabel("Core")
# ax2.set_ylabel("Core")

# plt.tight_layout()
# plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style
sns.set_style("darkgrid")
sns.set_context("poster")
sns.set_palette("colorblind")

batch_sizes = [8, 16, 32, 64, 128]
token_length_of_interest = 512

fig, axs = plt.subplots(len(batch_sizes), 1, figsize=(12, 10), sharey=True)

for idx, batch_size in enumerate(batch_sizes):
    specific_data = data[
        (data["batch_size"] == batch_size)
        & (data["tokens"] == token_length_of_interest)
        & (data["gpus"] == 1)
    ]

    total_power_specific = specific_data[
        ["socket0-package-power", "socket1-package-power"]
    ].sum(axis=1)

    sns.lineplot(
        data=total_power_specific.reset_index(drop=True),
        ax=axs[idx],
        color=sns.color_palette()[idx],
        linewidth=2,
    )
    axs[idx].set_title(f"Batch Size: {batch_size}", fontsize=16)
    axs[idx].grid(True, alpha=0.7)
    axs[idx].set_xlabel("Time (100ms intervals)", fontsize=14)
    axs[idx].set_ylabel("Total Power Consumption (W)", fontsize=14)

plt.suptitle(
    f"Total Power Draw for Token Length {token_length_of_interest} with Varying Batch Sizes",
    fontsize=20,
)
plt.tight_layout()
plt.show()


token_lengths = [128, 256, 512, 1024, 2048]
batch_size_of_interest = 32

fig, axs = plt.subplots(len(token_lengths), 1, figsize=(12, 10), sharey=True)

for idx, token_length in enumerate(token_lengths):
    specific_data = data[
        (data["tokens"] == token_length)
        & (data["batch_size"] == batch_size_of_interest)
        & (data["gpus"] == 1)
    ]

    total_power_specific = specific_data[
        ["socket0-package-power", "socket1-package-power"]
    ].sum(axis=1)

    sns.lineplot(
        data=total_power_specific.reset_index(drop=True),
        ax=axs[idx],
        color=sns.color_palette()[idx],
        linewidth=2,
    )
    axs[idx].set_title(f"Token Length: {token_length}", fontsize=16)
    axs[idx].grid(True, alpha=0.7)
    axs[idx].set_xlabel("Time (100ms intervals)", fontsize=14)
    axs[idx].set_ylabel("Total Power Consumption (W)", fontsize=14)

plt.suptitle(
    f"Total Power Draw for Batch Size {batch_size_of_interest} with Varying Token Length",
    fontsize=20,
)
plt.tight_layout()
plt.show()


batch_sizes = [8, 16, 32, 64, 128]
token_length_of_interest = 512

fig, axs = plt.subplots(len(batch_sizes), 1, figsize=(10, 6), sharey=True)

for idx, batch_size in enumerate(batch_sizes):
    # Filter data for each batch size
    specific_data = data[
        (data["batch_size"] == batch_size)
        & (data["tokens"] == token_length_of_interest)
        & (data["gpus"] == 1)
    ]

    # Calculate the total power for the specific token length and batch size
    total_power_specific = specific_data[
        ["socket0-package-power", "socket1-package-power"]
    ].sum(axis=1)

    # Plot
    axs[idx].plot(
        total_power_specific.reset_index(drop=True), label=f"Batch Size: {batch_size}"
    )
    axs[idx].set_title(f"Batch Size: {batch_size}")
    axs[idx].grid(True)

# Set common labels
for ax in axs:
    ax.set_xlabel("Time (100ms intervals)")
    ax.set_ylabel("Total Power Consumption (W)")

plt.suptitle(
    f"Total Power Draw for Token Length {token_length_of_interest} with Varying Batch Sizes"
)
plt.tight_layout()
plt.show()


token_lengths = [128, 256, 512, 1024, 2048]
batch_size_of_interest = 32

fig, axs = plt.subplots(len(token_lengths), 1, figsize=(10, 6), sharey=True)

for idx, token_length in enumerate(token_lengths):
    # Filter data for each batch size
    specific_data = data[
        (data["tokens"] == token_length)
        & (data["batch_size"] == batch_size_of_interest)
        & (data["gpus"] == 1)
    ]

    # Calculate the total power for the specific token length and batch size
    total_power_specific = specific_data[
        ["socket0-package-power", "socket1-package-power"]
    ].sum(axis=1)

    # Plot
    axs[idx].plot(
        total_power_specific.reset_index(drop=True),
        label=f"Token Length: {token_length}",
    )
    axs[idx].set_title(f"Token Length: {token_length}")
    axs[idx].grid(True)

# Set common labels
for ax in axs:
    ax.set_xlabel("Time (100ms intervals)")
    ax.set_ylabel("Total Power Consumption (W)")

plt.suptitle(
    f"Total Power Draw for Batch Size {batch_size_of_interest} with Varying Token Length"
)
plt.tight_layout()
plt.show()

# # Calculate the total power consumption for each timestamp
# total_power = data[["socket0-package-power", "socket1-package-power"]].sum(axis=1)

# # Convert the index to milliseconds assuming a sampling interval of 100ms
# runtime_ms = np.arange(len(total_power)) / 10

# # Plot the total power consumption over time with actual runtime
# plt.figure(figsize=(10, 5))
# plt.plot(runtime_ms, total_power)
# plt.xlabel("Runtime (s)")
# plt.ylabel("Total Power Consumption (W)")
# plt.title("Total CPU Energy Consumption over Runtime")
# plt.grid(True)
# plt.show()
# from datetime import datetime

# time_format = "%H:%M:%S:%f"
# start_time = datetime.strptime(data["Timestamp"].iloc[0], time_format)
# end_time = datetime.strptime(data["Timestamp"].iloc[-1], time_format)
# # duration = (end_time - start_time).total_seconds()

# # Calculate the mean total power consumption
# mean_total_power = total_power.mean()
# mean_total_energy = mean_total_power * duration

# print(f"Mean Total CPU Energy Consumption: {mean_total_energy:.2f} J")
