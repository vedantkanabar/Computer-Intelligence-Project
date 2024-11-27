import json
import numpy as np
from sklearn.preprocessing import StandardScaler


# A common function to extract features from the JSON file we have saved earlier.
# Inputs:
#   - fname: The file name to read from.
# Outputs:
#   - feature_array: An np array containing all feature vectors after normalization.
#   - feature_columns: Corresponding column names for feature vectors
#   - happiness_index_array: An np array containing all corresponding happiness_index
#     values for each feature.
def extract_features(fname):

    # Load the JSON data
    with open(fname, 'r') as file:
        data = json.load(file)

    feature_list = []
    happiness_index_list = []

    # Extract the feature columns
    feature_columns = data[0].get('features', {}).keys()

    # Loop through each object in the array and extract feature values
    for item in data:
        # Extract features as a dictionary
        features = item.get('features', {})
        
        # Get all values from the 'features' dictionary for feature_values and append to list
        feature_values = list(features.values())
        feature_list.append(feature_values)

        # Extract the happiness_index and append to list
        happiness_index = item.get('happiness_index', None)
        happiness_index_list.append(happiness_index)


    # Convert the list of feature values into a NumPy array
    feature_array = np.array(feature_list)
    happiness_index_array = np.array(happiness_index_list)

    # Normalize the data with a stanard scalar (z = (x - u) / s)
    scaler = StandardScaler()
    normalized_feature_array = scaler.fit_transform(feature_array)

    return normalized_feature_array, feature_columns, happiness_index_array


# A common function to extract features and labels from the JSON file we have saved earlier.
# Inputs:
#   - fname: The file name to read from.
# Outputs:
#   - feature_array: An np array containing all feature vectors after normalization.
#   - feature_label: Corresponding label for all feature vector
#   - feature_columns: Corresponding column names for feature vectors
#   - happiness_index_array: An np array containing all corresponding happiness_index
#     values for each feature.
def extract_features_label(fname):

    # Load the JSON data
    with open(fname, 'r') as file:
        data = json.load(file)

    feature_list = []
    feature_label_list = []
    happiness_index_list = []

    # Extract the feature columns
    feature_columns = data[0].get('features', {}).keys()

    # Loop through each object in the array and extract feature values
    for item in data:
        # Extract items as a dictionary
        features = item.get('features', {})
        labels = item.get('labels', {})
        
        # Get all values from the 'features' dictionary for feature_values and append to list
        feature_values = list(features.values())
        feature_list.append(feature_values)

        # Get all values from the 'features' dictionary for feature_values and append to list
        label_values = list(labels.values())
        feature_label_list.append(label_values)

        # Extract the happiness_index and append to list
        happiness_index = item.get('happiness_index', None)
        happiness_index_list.append(happiness_index)


    # Convert the list of feature values into a NumPy array
    feature_array = np.array(feature_list)
    feature_label = np.array(feature_label_list)
    happiness_index_array = np.array(happiness_index_list)

    # Normalize the data with a stanard scalar (z = (x - u) / s)
    scaler = StandardScaler()
    normalized_feature_array = scaler.fit_transform(feature_array)

    return normalized_feature_array, feature_label, feature_columns, happiness_index_array