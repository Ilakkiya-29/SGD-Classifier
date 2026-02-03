# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries.
2. Load the Iris dataset.
3. Split the dataset into training and testing data.
4. Create the SGD Classifier model.
5. Train the model using training data.
6. Predict the species using test data.
7. Display the predicted output.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ilakkiya K
RegisterNumber: 212225040130
*/# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset
data = pd.read_csv("Placement_Data_Full_Class.csv")

# 3️⃣ Separate Features and Target
X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]   # Placed / Not Placed

# 4️⃣ Convert Categorical to Numerical (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# 8️⃣ Train Model
model.fit(X_train, y_train)

# 9️⃣ Predict
y_pred = model.predict(X_test)

# 🔟 Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1️⃣1️⃣ Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 1️⃣2️⃣ Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 1️⃣3️⃣ Predict for New Student (Example)
new_student = [[67, 91, 58, 0, 1, 1, 1, 0, 0, 88, 1, 67]]
new_student = scaler.transform(new_student)

pred = model.predict(new_student)
print("\nPredicted Status:", pred[0])
```

## Output:
![WhatsApp Image 2026-02-03 at 9 38 55 AM](https://github.com/user-attachments/assets/e1755025-ed88-4e51-bcb6-6308dc1052e5)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
