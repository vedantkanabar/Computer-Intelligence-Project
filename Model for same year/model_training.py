import json
from scipy.stats import randint
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from utils import extract_features

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features('../Combining Data and Feature creation/feature.json')

# Split the data into training and testing sets (80% train, 20% test)
feature_train, feature_test, happiness_index_train, happiness_index_test = train_test_split(feature_array, happiness_index_array, test_size=0.2)


# Set model to Random Forest
model = RandomForestRegressor(
    bootstrap=True, 
    max_depth=30,
    min_samples_leaf=1, 
    min_samples_split=2, 
    n_estimators=196
    )

print("Now training model...")
model.fit(feature_train, happiness_index_train)

happiness_index_pred = model.predict(feature_test)

# Calculate r2 and MSE for regression evaluation
r2 = r2_score(happiness_index_test, happiness_index_pred)
mse = mean_squared_error(happiness_index_test, happiness_index_pred)
rmse = root_mean_squared_error(happiness_index_test, happiness_index_pred)
mae = mean_absolute_error(happiness_index_test, happiness_index_pred)

# Output the accuracy
print(f"r2: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

feature_importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_columns,  # assuming X_train is a DataFrame with column names
    'importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Select the top N features
n = 10  # You can adjust this number based on how many top features you want
top_n_features = feature_importance_df.head(n)

# Display the top N features
print(top_n_features)