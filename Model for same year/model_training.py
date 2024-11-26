import json
from scipy.stats import randint
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

# Load the JSON data (Assuming it's in a file called 'data.json')
with open('../Combining Data and Feature creation/feature.json', 'r') as file:
    data = json.load(file)

all_feature_values = []
target_list = []

feature_columns = data[0].get('features', {}).keys()

print(feature_columns)

# Loop through each object in the array and extract feature values
for item in data:
    # Extract features dictionary for the current object
    features = item.get('features', {})
    
    # Get all values from the 'features' dictionary
    feature_values = list(features.values())
    
    # Append the feature values to the list
    all_feature_values.append(feature_values)

    # Extract the target variable (happiness_index)
    happiness_index = item.get('happiness_index', None)

    target_list.append(happiness_index)


# Convert the list of feature values into a NumPy array
feature_array = np.array(all_feature_values)
target_array = np.array(target_list)

print(feature_array.shape)

model = RandomForestRegressor()

# scoring_metrics = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']

# Perform K-Fold Cross-Validation (using 5 folds in this case)
# cv_results = cross_validate(model, feature_array, target_array, cv=10, scoring=scoring_metrics)

# Print the cross-validation scores and their mean
# print(f"Cross-Validation Mean Squared Errors for each fold: {cv_scores}")
# print(f"Mean of Cross-Validation MSE: {cv_scores.mean()}")

# for metric in scoring_metrics:
#     print(f"{metric} for each fold: {cv_results['test_' + metric]}")
#     print(f"Mean {metric}: {cv_results['test_' + metric].mean()}")
#     print(f"Standard deviation of {metric}: {cv_results['test_' + metric].std()}")
#     print("-" * 50)c

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
random_search.fit(feature_array, target_array)
print("Best parameters found: ", random_search.best_params_)
best_model = random_search.best_estimator_
print(best_model)


print("Now training model...")

model.fit(feature_array, target_array)

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