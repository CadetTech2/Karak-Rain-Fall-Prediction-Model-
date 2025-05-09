import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset - Replace with real data later from Pakistan Meteorological Dept.
data = """Month\tTemperature\tHumidity\tRainfall
1\t14\t65\t25
2\t17\t60\t20
3\t23\t55\t15
4\t30\t45\t10
5\t35\t35\t5
6\t38\t30\t3
7\t40\t25\t2
8\t39\t28\t3
9\t33\t40\t7
10\t28\t50\t15
11\t22\t55\t18
12\t16\t60\t22
"""

# Create DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), sep='\t')

# Split the data into features (X) and target (y)
X = df[['Month', 'Temperature', 'Humidity']]  # Features
y = df['Rainfall']  # Target

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation: Calculate Mean Squared Error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score (Accuracy of Fit): {r2:.2f}")

# Visualize the actual data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature'], df['Rainfall'], color='blue', label='Actual Data')

# Plot the regression line
temp_range = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 100)
humidity_mean = df['Humidity'].mean()
month_mean = df['Month'].mean()

# Prepare X for prediction
X_pred = np.column_stack([np.full(100, month_mean), temp_range, np.full(100, humidity_mean)])
y_pred_line = model.predict(X_pred)

plt.plot(temp_range, y_pred_line, color='red', linewidth=2, label='Regression Line (Model Prediction)')

plt.title('Rainfall Prediction in Karak vs Temperature', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Rainfall (mm)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# User Input Prediction (Interactive)
temp_input = float(input("\n🌡️ Enter Temperature (°C): "))
humidity_input = float(input("💧 Enter Humidity (%): "))

# Predict Rainfall using the model
predicted_rainfall = model.predict([[month_mean, temp_input, humidity_input]])[0]
print(f"\n🌧️ Predicted Rainfall for Temperature {temp_input}°C and Humidity {humidity_input}% is **{predicted_rainfall:.2f} mm**")
