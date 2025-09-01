import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load first, then fill missing
df = pd.read_csv("datasets/iris.csv")
df = df.fillna(df.mean(numeric_only=True))

# Features & Labels
X, y = df.drop("species", axis=1), df["species"]
le = LabelEncoder()
y = le.fit_transform(y)

# Train-Test Split + Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel="linear").fit(X_train, y_train)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# Predict new flower
# [Sepal Length , Sepal Width , Petal Length , Petal Width]
new_flower = [[5, 4, 4.5, 5.9]]
pred = model.predict(new_flower)[0]   # numeric label (0, 1, or 2)
name = le.inverse_transform([pred])[0]  # species name

print(f"Predicted: {pred+1} - {name}")

# -----------------------------
# Plotting with Legends
# -----------------------------
plt.figure(figsize=(8, 6))

# Use first 2 features (sepal length vs sepal width) for visualization
for i, species in enumerate(le.classes_):
    plt.scatter(
        X_test[y_test == i].iloc[:, 0], 
        X_test[y_test == i].iloc[:, 1],
        label=species
    )

# Plot the new flower point
plt.scatter(new_flower[0][0], new_flower[0][1], 
            color="black", marker="X", s=120, label="New Flower")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification (Test Data)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
