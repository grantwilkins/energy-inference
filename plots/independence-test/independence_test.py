import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# input_data = pd.read_csv("all-input-stats.csv")
# output_data = pd.read_csv("all-output-stats.csv")

# input_data = input_data[input_data["Phase"].str.contains("inference")]
# output_data = output_data[output_data["Phase"].str.contains("inference")]
# input_data = input_data[~input_data["Phase"].str.contains("-0")]
# output_data = output_data[~output_data["Phase"].str.contains("-0")]

# data = pd.concat(
#     [input_data, output_data],
#     ignore_index=True,
# )
# data = pd.read_csv("all-stats-raw.csv")
data = pd.read_csv("Llama-2-7b-chat-hf-M1-Pro.csv")
data = data[data["Phase"].str.contains("inference")]
data["Total Energy (J)"] = data["GPU Energy (J)"] + data["CPU Energy (J)"]
# data["Total Energy (J)"] = data["GPU Energy (J)"] / 1e3
data["Energy per Token (J/tokens)"] = (
    data["Total Energy (J)"] / data["Total Number of Tokens"]
)
data.fillna(0, inplace=True)

# data = data[data["System"] != "Palmetto Intel+A100"]
# data = data[data["System"] == "Swing AMD+A100"]
# data_m1pro = data[data["System"] == "M1-Pro"]
# data_swing = data[data["System"] == "Swing AMD+A100"]
# data_palmetto = data[data["System"] == "Palmetto Intel+V100"]

# data = data[data["Model"] == "Llama-2 (7B)"]

# print(data)
# # For Input data: assuming 'runtime' as dependent and 'input_tokens' as independent
# # model = ols(
# #     'Q("Total Energy (J)") ~ C(Q("Number of Input Tokens")) + C(Q("Number of Output Tokens")) + C(Q("Number of Input Tokens")):C(Q("Number of Output Tokens"))',
# #     data=data,
# # ).fit()
# model = ols(
#     'Q("Total Energy (J)") ~ C(Q("Number of Output Tokens")) + C(Q("Number of Input Tokens")) + C(Q("Number of Output Tokens")):C(Q("Number of Input Tokens"))',
#     data=data,
# ).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)


# Calculate Pearson correlation and p-value for input tokens and energy
# corr_input, p_value_input = pearsonr(
#     data_m1pro["Number of Input Tokens"], data_m1pro["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Input Tokens and Energy) on M1-Pro: {corr_input}, P-value: {p_value_input}"
# )

# # Calculate Pearson correlation and p-value for output tokens and energy
# corr_output, p_value_output = pearsonr(
#     data_m1pro["Number of Output Tokens"], data_m1pro["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Output Tokens and Energy) on M1-Pro: {corr_output}, P-value: {p_value_output}"
# )

# # Calculate Pearson correlation and p-value for input tokens and energy
# corr_input, p_value_input = pearsonr(
#     data_swing["Number of Input Tokens"], data_swing["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Input Tokens and Energy) on Swing: {corr_input}, P-value: {p_value_input}"
# )

# # Calculate Pearson correlation and p-value for output tokens and energy
# corr_output, p_value_output = pearsonr(
#     data_swing["Number of Output Tokens"], data_swing["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Output Tokens and Energy) on Swing: {corr_output}, P-value: {p_value_output}"
# )

# # Calculate Pearson correlation and p-value for input tokens and energy
# corr_input, p_value_input = pearsonr(
#     data_palmetto["Number of Input Tokens"], data_palmetto["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Input Tokens and Energy) on Palmetto: {corr_input}, P-value: {p_value_input}"
# )

# # Calculate Pearson correlation and p-value for output tokens and energy
# corr_output, p_value_output = pearsonr(
#     data_palmetto["Number of Output Tokens"], data_palmetto["Total Energy (J)"]
# )
# print(
#     f"Pearson Correlation (Output Tokens and Energy) on Palmetto: {corr_output}, P-value: {p_value_output}"
# )

# df_falcon = data[data["Model"] == "falcon-7b"]
# df_mistral = data[data["Model"] == "Mistral-7B-v0.1"]

model = sm.OLS.from_formula(
    'Q("Total Energy (J)") ~ C(Q("Number of Output Tokens")) + C(Q("Number of Input Tokens"))  + C(Q("Number of Output Tokens")):C(Q("Number of Input Tokens"))',
    data=data,
)
result = model.fit()
print(sm.stats.anova_lm(result, typ=2))

model = sm.OLS.from_formula(
    'Q("Runtime (s)") ~  C(Q("Number of Output Tokens")) +C(Q("Number of Input Tokens")) + C(Q("Number of Input Tokens")):C(Q("Number of Output Tokens"))',
    data=data,
)
result = model.fit()
print(sm.stats.anova_lm(result, typ=2))

sns.set(style="whitegrid", context="talk", palette="tab10")
plt.figure(figsize=(15, 6))
# sns.set_palette("colorblind")
sns.lineplot(
    x="Number of Output Tokens",
    y="Total Energy (J)",
    hue="Number of Input Tokens",
    # style="Model",
    palette="tab10",
    data=data,
    hue_order=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
)
plt.legend(
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
    #     ],
    title="Input Tokens",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(which="minor", color="black", linestyle=":", linewidth=0.5, alpha=0.3)

plt.savefig("input-tokens-total-energy-line.pdf", bbox_inches="tight")

# data = data[data["Model"] == "falcon-7b"]
# data = data[data["Model"] == "falcon-40b"]
# data = data[data["Model"] == "Mixtral-8x7B-v0.1"]
# data = data[data["Model"] == "Mistral-7B-v0.1"]
# data = data[data["Model"] == "Llama-2-7b-chat-hf"]
# data = data[data["Model"] == "Llama-2-13b-chat-hf"]
# data = data[data["Model"] == "Llama-2-70b-chat-hf"]
# data = data[data["Model"] == "Llama-2 (7B)"]

data["Interaction"] = data["Number of Input Tokens"] * data["Number of Output Tokens"]
data["Square of Output Tokens"] = data["Number of Output Tokens"] ** 2
X = data[
    [
        "Number of Input Tokens",
        "Number of Output Tokens",
        "Interaction",
    ]
]
# y = data["Runtime (s)"]
y = data["Total Energy (J)"]
# y = data["Energy per Token (J/tokens)"]
model = sm.OLS(y, X).fit()
print(model.summary())
print(model.summary2().tables[1])

# # Prepare the data for prediction
# # Define powers of two from 8 to 2048 for input and output tokens
# input_tokens = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# output_tokens = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# # Prepare data for prediction
# predict_data = {
#     "Number of Input Tokens": [i for i in input_tokens for _ in output_tokens],
#     "Number of Output Tokens": [o for _ in input_tokens for o in output_tokens],
#     "Interaction": [i * o for i in input_tokens for o in output_tokens],
# }

# # Convert the dictionary to a DataFrame
# predict_df = pd.DataFrame(predict_data)

# # Make the prediction using the OLS model
# predictions = model.predict(predict_df)

# # Add predictions back to DataFrame
# predict_df["Predicted Total Energy (J)"] = predictions

# # Plotting
# plt.figure(figsize=(10, 8))
# sns.lineplot(
#     data=predict_df,
#     x="Number of Input Tokens",
#     y="Predicted Total Energy (J)",
#     hue="Number of Output Tokens",
#     # palette=sns.color_palette("viridis", as_cmap=True),
#     marker="o",
# )
# sns.lineplot(
#     data=data,
#     x="Number of Input Tokens",
#     y="Total Energy (J)",
#     hue="Number of Output Tokens",
#     palette="tab10",
#     marker="o",
#     legend=False,
# )
# # plt.xscale("log", base=2)
# # plt.yscale("log")
# plt.title("Predicted Energy per Token for Various Input and Output Tokens")
# plt.xlabel("Number of Input Tokens")
# plt.ylabel("Predicted Energy per Token (J/tokens)")
# plt.legend(title="Number of Output Tokens")
# plt.grid(True)
# plt.show()
