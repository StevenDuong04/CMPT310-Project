import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

data = pd.read_csv("../student-scores.csv")  # loads dataset

# inspect structure
print(data.head())
print(data.info())

# 3. Decide on target + features
# Example: predicting 'JobTitle' (replace with actual column name)
X = data.drop("JobTitle", axis=1)
y = data["JobTitle"]

# 4. Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize model
model = DecisionTreeClassifier(random_state=42)

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))