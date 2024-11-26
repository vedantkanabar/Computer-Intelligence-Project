import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from utils import extract_features

# Extract features from json file
feature_array, feature_columns, happiness_index_array = extract_features('../Combining Data and Feature creation/feature.json')

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Regression": SVR(),
}

# Store the result for printing purposes
evalResult = []

for name, model in models.items():

    print(f"Evaluating {name} ...")
    
    r2_scores = cross_val_score(model, feature_array, happiness_index_array, cv=10, scoring='r2')
    mse_scores = -cross_val_score(model, feature_array, happiness_index_array, cv=10, scoring='neg_mean_squared_error')

    evalResult.append({
        "Model": name,
        "Average R2 Score": r2_scores.mean(),
        "Standard Deviation R2": r2_scores.std(),
        "Min R2": r2_scores.min(),
        "Max R2": r2_scores.max(),
        "Average MSE": mse_scores.mean(),
        "Standard Deviation MSE": mse_scores.std(),
        "Min MSE": mse_scores.min(),
        "Max MSE": mse_scores.max(),
    })

    print("Finished Evaluation")

results = pd.DataFrame(evalResult)
print("\n10-Fold Cross-Validation Results:")
print(results)

# Find the best model based on Average R2 Score
best_model = results.loc[results["Average R2 Score"].idxmax()]
print(f"\nBest model based on Average R2 Score: {best_model['Model']}")

# Find the best model based on Average MSE score
best_model = results.loc[results["Average MSE"].idxmin()]
print(f"\nBest model based on Average MSE Score: {best_model['Model']}")



