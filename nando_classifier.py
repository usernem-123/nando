import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load first, then fill missing
df = pd.read_csv("iris.csv")
df = df.fillna(df.mean(numeric_only=True))

# Features & Labels
X, y = df.drop("species", axis=1), df["species"]
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split + Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel="linear").fit(X_train, y_train)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# Visualization
for cls in set(y):
    plt.scatter(X[y==cls].iloc[:,0], X[y==cls].iloc[:,1], label=le.inverse_transform([cls])[0])
plt.xlabel("Sepal Length"); plt.ylabel("Sepal Width"); plt.legend(); plt.show()

# Predict new flower
print("Predicted:", le.inverse_transform(model.predict([[9, 2, 2.1, 1.5]]))[0])
