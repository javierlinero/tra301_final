#!/usr/bin/env python3
"""
evaluate_contrib.py

Compute the fraction of gold <CONTRIBUTION> sentences correctly
predicted by your filter.
"""
from collections import defaultdict
import pandas as pd
import re
import pprint

gold_pth = r"C:\Users\JL211\Downloads\evaluation_results_tagged.csv"
pred_pth = r"C:\Users\JL211\Downloads\evaluation_results_csv_filtered.csv"

gold = pd.read_csv(gold_pth)
pred = pd.read_csv(pred_pth)

gold_dict = dict(zip(gold["filename"], gold["input_text"].fillna("")))
pred_dict = dict(zip(pred["filename"], pred["input_text"].fillna("")))

converted = defaultdict(list)
for fn, text in gold_dict.items():
    for word in text.split():
        if "<BACKGROUND>" in word:
            converted[fn].append(0)
        elif "<CONTRIBUTION>" in word:
            converted[fn].append(1)

converted_filtered = defaultdict(list)
for fn, text in pred_dict.items():
    for word in text.split():
        if "<BACKGROUND>" in word:
            converted_filtered[fn].append(0)
        elif "<CONTRIBUTION>" in word:
            converted_filtered[fn].append(1)

print("\nIDEAL")
for key, val in sorted(converted.items()):
    print(f"{len(val)} - {key}: {val}")

print("\nGENERATED:")

for key, val in sorted(converted_filtered.items()):
    print(f"{len(val)} - {key}: {val}")


for key in sorted(converted.keys()):
    score = 0
    length = len(converted[key])
    for idx in range(length):
        if converted[key][idx] == converted_filtered[key][idx]:
            score += 1

    print(f"Score for {key}: {score / length}")

