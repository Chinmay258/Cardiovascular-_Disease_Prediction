import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

file_path = 'cardio_train.csv'
data = pd.read_csv(file_path, delimiter=';')

data = data.drop_duplicates()

X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Correlation Matrix:")
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Correlation matrix plot saved.")

print("Target Distribution:")
sns.countplot(x=y)
plt.title('Distribution of Target Variable (Cardio)')
plt.savefig('target_distribution.png')
print("Target distribution plot saved.")

print("Feature Boxplot:")
sns.boxplot(data=X)
plt.title('Boxplot of Features')
plt.savefig('feature_boxplot.png')
print("Feature boxplot saved.")

models = {
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

print("Training and evaluating models:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    results[name] = {"Accuracy": acc, "ROC-AUC": roc}

    print(f"Model: {name}")
    print("Accuracy:", acc)
    if roc is not None:
        print("ROC-AUC:", roc)
    print(classification_report(y_test, y_pred))
    print("\n")

best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
best_model = models[best_model_name]

print(f"The best model is: {best_model_name}")

joblib.dump(best_model, 'heart_disease_model.pkl')
print("Best model saved.")

model = joblib.load('heart_disease_model.pkl')

new_data = pd.DataFrame([[45, 1, 160, 70, 120, 80, 1, 1, 0, 0, 1]], columns=X.columns)
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")

