from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from utils import extract_features

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

# Set model to random forest
model = RandomForestRegressor()

# Hyperparameter tuning
print("Hyperparameter tuning...")

param_dist = {
    "n_estimators": randint(50, 200),         # Random search between 50 and 200
    "max_depth": [None, 10, 20, 30],          # Max depth of trees
    "min_samples_split": randint(2, 10),      # Min samples to split
    "min_samples_leaf": randint(1, 5),        # Min samples at leaf node
    "bootstrap": [True, False]                # Whether to use bootstrap sampling
}

# Perform rendomized search for hyper parameters
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring="neg_mean_squared_error")
random_search.fit(feature_array, happiness_index_array)

# Print Results for best parameters found
print("Best parameters found: ", random_search.best_params_)
