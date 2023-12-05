"""Combines the metrics from the JSON file into a single average for each metric.
"""

import json
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

# File path for the JSON file, taken from command-line argument
file_path = sys.argv[1]

# Initialize variables to store the sum of scores
sum_rouge_1 = 0
sum_rouge_L = 0
sum_bleu = 0
num_records = 0

# Read and parse the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

    # Iterate over each record
    for record in data:
        # Check if elements in 'rouge 1' and 'rouge L' are numeric and then sum
        if all(isinstance(item, (int, float)) for item in record["rouge 1"]):
            sum_rouge_1 += sum(record["rouge 1"]) / len(record["rouge 1"])
        if all(isinstance(item, (int, float)) for item in record["rouge L"]):
            sum_rouge_L += sum(record["rouge L"]) / len(record["rouge L"])

        # Assuming 'bleu' is always a number
        sum_bleu += record["bleu"]
        num_records += 1

# Calculate the average for each score
avg_rouge_1 = sum_rouge_1 / num_records
avg_rouge_L = sum_rouge_L / num_records
avg_bleu = sum_bleu / num_records

# Print the averages
print(f"Average Rouge 1: {avg_rouge_1}")
print(f"Average Rouge L: {avg_rouge_L}")
print(f"Average BLEU: {avg_bleu}")
print(avg_rouge_1, avg_rouge_L, avg_bleu)
