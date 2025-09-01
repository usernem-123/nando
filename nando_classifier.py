import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel="linear").fit(X_train, y_train)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# Predict new flower
new_flower = [[5, 8, 4.5, 5.9]]
pred = model.predict(new_flower)[0]   # numeric label (0, 1, or 2)
name = le.inverse_transform([pred])[0]  # species name

# Add +1 para mahimong 1/2/3 imbis 0/1/2
print(f"Predicted: {pred+1} - {name}")

