import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import extract_features_label

EU_Countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

feature_array, feature_label, feature_columns, happiness_index_array = extract_features_label('../Combining Data and Feature creation/feature.json')

# Scale the feature data
scaler = StandardScaler()
feature_scaled = scaler.fit_transform(feature_array)

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(feature_scaled)

# Plot the PCA results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=happiness_index_array, cmap='viridis', edgecolor='k', s=100)

# Add labels and title
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA - 2D Scatterplot of Reduced Features')

# Show a colorbar to represent happiness index
plt.colorbar(scatter, label='Happiness Index')

# Display the plot
plt.show()

