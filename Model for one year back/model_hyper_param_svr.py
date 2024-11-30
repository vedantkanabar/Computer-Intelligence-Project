from scipy.stats import randint
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from utils import extract_features
from scipy.stats import uniform

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

model = SVR()

# Hyperparameter tuning
print("Hyperparameter tuning...")

param_dist = {
    'C': uniform(0.1, 10),                           # Regularization parameter, sampled from 0.1 to 10
    'epsilon': uniform(0.01, 1),                     # Epsilon-tube width, sampled from 0.01 to 1
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types for SVR
    'degree': [2, 3, 4, 5],                          # Polynomial degree, used if kernel is 'poly'
    'gamma': ['scale', 'auto']                       # Kernel coefficient options
}

# Perform rendomized search for hyper parameters
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring="neg_mean_squared_error")
random_search.fit(feature_array, happiness_index_array)

# Print Results for best parameters found
print("Best parameters found: ", random_search.best_params_)
print(random_search.best_estimator_)

