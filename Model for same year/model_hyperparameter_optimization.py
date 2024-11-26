import json
from scipy.stats import randint
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from utils import extract_features

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features('../Combining Data and Feature creation/feature.json')

# Set model to random forest
model = RandomForestRegressor()

# Hyperparameter tuning
print("Hyperparameter tuning...")

param_dist = {
    'n_estimators': randint(50, 200),        # Random search between 50 and 200
    'max_depth': [None, 10, 20, 30],          # Max depth of trees
    'min_samples_split': randint(2, 10),      # Min samples to split
    'min_samples_leaf': randint(1, 5),        # Min samples at leaf node
    'bootstrap': [True, False]                # Whether to use bootstrap sampling
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(feature_array, happiness_index_array)
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_
print(best_model)


# print("Now training model...")

# model.fit(feature_array, happiness_index_array)

# feature_importances = model.feature_importances_

# feature_importance_df = pd.DataFrame({
#     'feature': feature_columns,  # assuming X_train is a DataFrame with column names
#     'importance': feature_importances
# })

# feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# # Select the top N features
# n = 10  # You can adjust this number based on how many top features you want
# top_n_features = feature_importance_df.head(n)

# # Display the top N features
# print(top_n_features)