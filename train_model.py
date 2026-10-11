import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

print("Dataset loading...")
df = pd.read_csv("dataset/software_projects.csv")
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Training Model...")
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("SHAP Graph banvatoy...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
print("Graph saved: shap_summary.png - HA KHUP MAHATVACHA AAHE!")
plt.show()