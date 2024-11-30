import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.metrics import PredictionErrorDisplay
from sklearn.inspection import permutation_importance
from utils import extract_features

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

# Split the data into training and testing sets (80% train, 20% test)
feature_train, feature_test, happiness_index_train, happiness_index_test = train_test_split(feature_array, happiness_index_array, test_size=0.2)


# Set model to Random Forest
model = RandomForestRegressor(
    bootstrap=True, 
    max_depth=30,
    min_samples_leaf=2, 
    min_samples_split=4, 
    n_estimators=151
    )

# Train the model
print("Now training Random Forest Regression model...")
model.fit(feature_train, happiness_index_train)

# Test the model
happiness_index_pred = model.predict(feature_test)

# Calculate r2, MSE, R-MSE, ASE for regression evaluation
r2 = r2_score(happiness_index_test, happiness_index_pred)
mse = mean_squared_error(happiness_index_test, happiness_index_pred)
rmse = root_mean_squared_error(happiness_index_test, happiness_index_pred)
mae = mean_absolute_error(happiness_index_test, happiness_index_pred)

# Output the results
print()
print("Random Forest Regression model results:")
print(f"r2: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Create Actual vs Predicted plot
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
PredictionErrorDisplay.from_predictions(
    y_true=happiness_index_test,
    y_pred=happiness_index_pred,
    kind="actual_vs_predicted",
    ax=axs[0],
)
axs[0].set_title("Actual vs. Predicted values")

# Create Residual vs Predicted plot
PredictionErrorDisplay.from_predictions(
    y_true=happiness_index_test,
    y_pred=happiness_index_pred,
    kind="residual_vs_predicted",
    ax=axs[1],
)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting Random Forest Regression predictions")

# Save both error plots
plt.tight_layout()
plt.savefig("Random Forest Regression Residual plot")

# Get top features
# feature_importances = model.feature_importances_
feature_importances = permutation_importance(model, feature_test, happiness_index_test, n_repeats=10, n_jobs=-1)

# Create datafreame to display the info
feature_importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": feature_importances.importances_mean
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

# Select the top 10 features
n = 10
top_n_features = feature_importance_df.head(n)

# Display the top N features
print()
print("Random Forest Regression model most important features:")
print(top_n_features)

# Get indexes of 10 features
sorted_idx = np.argsort(feature_importances.importances_mean)[-10:]

# Exract top features and their names through the indexes
top_features = np.array(feature_columns)[sorted_idx]
top_importances = feature_importances.importances_mean[sorted_idx]

# Wrap column names for plots
wrapped_features = ["\n".join(textwrap.wrap(feature, width=30)) for feature in top_features]

# Create and Save plot
pos = np.arange(len(top_features)) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.barh(pos, top_importances, align="center")
plt.yticks(pos, wrapped_features)
plt.title("Feature Importance for Random Forest Regression through feature permuatation")
fig.tight_layout()
plt.savefig("Random Forest Regression Important vectors graph")