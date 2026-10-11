import pandas as pd
import numpy as np
import os

# dataset folder nasel tar banav
os.makedirs("dataset", exist_ok=True)

np.random.seed(42)
n = 200 # 200 software projects

data = {
    "team_size": np.random.randint(3, 15, n),
    "code_complexity": np.random.randint(1, 10, n),
    "bug_count": np.random.randint(0, 50, n),
    "req_changes": np.random.randint(0, 20, n),
    "deadline_pressure": np.random.randint(1, 10, n),
}

df = pd.DataFrame(data)

# Risk logic - simple
def get_risk(row):
    score = row["bug_count"]*0.5 + row["req_changes"]*2 + row["deadline_pressure"]*3
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    else:
        return "High"

df["risk_level"] = df.apply(get_risk, axis=1)
df.to_csv("dataset/software_projects.csv", index=False)
print("Dataset banla! dataset/software_projects.csv saved")