import re

log_path = r"ab.txt"

values = []

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"Objective value:\s*([0-9.+-eE]+)", line)
        if match:
            values.append(round(float(match.group(1)), 3))

print(values)

print(len(values))
