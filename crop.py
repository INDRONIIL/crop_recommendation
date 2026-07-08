import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\Crop_recommendation.csv")
pd.set_option('display.max_rows',None)
print(df)

# Features and Target
X = df.drop("label", axis=1)
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_data = pd.DataFrame({
    "N": [90],
    "P": [42],
    "K": [43],
    "temperature": [20.88],
    "humidity": [82.00],
    "ph": [6.50],
    "rainfall": [202.94]
})
prediction = model.predict(new_data)
print("Recommended Crop:", prediction[0])