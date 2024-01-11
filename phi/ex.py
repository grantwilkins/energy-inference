import re

def extract_last_number_from_specific_lines(lines, keywords):
    extracted_numbers = {}
    pattern = r'\s+(\d+\.?\d*)$'

    for line in lines:
        # Check if the line starts with any of the specified keywords
        if any(line.startswith(keyword) for keyword in keywords):
            match = re.search(pattern, line)
            if match:
                extracted_numbers[line.split()[0]] = float(match.group(1))  # Convert to float
            else:
                print(f"No number found at the end of the line for {line.split()[0]}.")

    return extracted_numbers

# Example lines
lines = [
    "python3.11                         42448  1247.69   83.12  0.00    0.00               25.79   0.00              2706.29",
    "ALL_TASKS                          -2     3190.78   71.27  533.41  0.00               3157.44 0.00              4476.72"
]

# Keywords to look for
keywords = ["python3.11", "ALL_TASKS"]

# Extract and print the last number from specific lines
extracted_numbers = extract_last_number_from_specific_lines(lines, keywords)
for key, value in extracted_numbers.items():
    print(f"Last number in '{key}' line: {value}")

