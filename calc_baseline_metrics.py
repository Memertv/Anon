import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# Load the cleaned dataset
file_path = "FUNAAB_ICTREC_Helpdesk_Cleaned_Orange.csv"
df = pd.read_csv(file_path)

# Define features (X) and target (y)
features = ['User_Type', 'Unit_Dept', 'Issue_Category', 'Priority']
X = df[features]
y_label = df['Resolution_Category']

# Map to 1 (Medium/Delayed) and 0 (Fast) to match the Advanced model
y = np.where(y_label == 'Medium (8 hrs - 1 week)', 1, 0)

# One-hot encode categorical features for the model
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"Accuracy (CA): {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"MCC: {matthews_corrcoef(y_test, y_pred):.3f}")
