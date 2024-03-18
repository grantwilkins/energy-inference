import os
import pandas as pd


def extract_metadata(file_path):
    # Extract metadata from the file path
    parts = file_path.split("/")
    print(parts)
    model = parts[1]
    date = parts[2]
    system = parts[3]
    gpus = parts[4].split("-")[1]
    tokens = parts[4].split("-")[3]
    batch_size = parts[4].split("-")[5]

    return {
        "model": model,
        "date": date,
        "system": system,
        "gpus": int(gpus),
        "tokens": int(tokens),
        "batch_size": int(batch_size),
    }


def process_timechart_files(base_directory):
    dataframes = []

    # Traverse through all subdirectories and find timechart.csv files
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file == "timechart.csv":
                file_path = os.path.join(root, file)

                # Extract metadata from the file path
                metadata = extract_metadata(file_path)

                # Read the timechart.csv file into a DataFrame
                df = pd.read_csv(file_path, sep=",", skiprows=150)

                # Add metadata columns to the DataFrame
                for key, value in metadata.items():
                    df[key] = value

                dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


# Specify the base directory where the power records are stored
base_directory = "./mistral-7b"
model_name = base_directory.split("/")[1]

# Process all timechart.csv files and create a combined DataFrame
power_records_df = process_timechart_files(base_directory)

# Save the combined DataFrame to a CSV file
power_records_df.to_csv(f"{model_name}_power_records.csv", index=False)
