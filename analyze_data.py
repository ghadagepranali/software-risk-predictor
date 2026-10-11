import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")
df = pd.read_csv("dataset/software_projects.csv")
print(df.head())

print("\nRisk Count:")
print(df['risk_level'].value_counts())

plt.figure()
df['risk_level'].value_counts().plot(kind='bar', color=['green','orange','red'])
plt.title("Software Project Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Projects")
plt.savefig("risk_distribution.png")
print("Graph saved: risk_distribution.png")
plt.show()