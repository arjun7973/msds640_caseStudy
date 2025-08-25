import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (prefer cleaned version if available)
try:
    df = pd.read_csv("cleaned_items.csv")
except FileNotFoundError:
    df = pd.read_csv("20191226-items.csv")

# Drop rows with missing values for modeling
df = df.dropna(subset=['price', 'brand'])

# Encode categorical brand as numeric
df['brand_encoded'] = df['brand'].astype('category').cat.codes

# Features and target
X = df[['brand_encoded']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Predictive Modeling Results ===")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
