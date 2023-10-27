# $$$$$$$$$  FEATURE ENGINEERING  $$$$$$$


import pandas as pd

# Load the dataset
data = pd.read_csv('Microsoft_Stock_Price.csv')

# Check and preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Feature Engineering: Create additional features if needed

# For example, you can create lag features for past N days
lag_days = 5
for i in range(1, lag_days + 1):
    data[f'Price_Lag_{i}'] = data['Close'].shift(i)

# Drop rows with missing values
data.dropna(inplace=True)




# $$$$$$$$$$    MODEL TRAINING   $$$$$$$

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Split the data into features and target variable
X = data.drop(['Close'], axis=1)
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model (Random Forest in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# $$$$$$ EVALUATION $$$$$$

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predicted vs. actual prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
