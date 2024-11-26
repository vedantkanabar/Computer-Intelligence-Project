import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#Loading the data
with open('../Combining Data and Feature creation/feature.json', 'r') as file:
    data = json.load(file)

all_feature_values = []
target_list = []

feature_columns = data[0].get('features', {}).keys()

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

#Models to evaluate
models = {
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor()
}

#Store the result for printing purposes
evalResult = []

for name, model in models.items():

    print(f"Evaluating {name}...")
    
    r2_scores = cross_val_score(model, feature_array, target_array, cv=10, scoring='r2')
    mse_scores = -cross_val_score(model, feature_array, target_array, cv=10, scoring='neg_mean_squared_error')

    evalResult.append({
        "Model": name,
        "Average R2 Score": r2_scores.mean(),
        "Standard Deviation R2": r2_scores.std(),
        "Average MSE": mse_scores.mean(),
        "Standard Deviation MSE": mse_scores.std()
    })

results = pd.DataFrame(evalResult)
print("\n10-Fold Cross-Validation Results:")
print(results)

# Find the best model based on Average R2 Score
best_model = results.loc[results["Average R2 Score"].idxmax()]
print(f"\nBest model based on Average R2 Score: {best_model['Model']}")



