from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import extract_features

feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

# Apply PCA for dimensionality reduction to 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(feature_array)

# Plot the PCA results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=happiness_index_array, cmap="viridis", edgecolor="k", s=50)

ax.set_title("3D Visualization of PCA Reduced Features")
ax.set_xlabel("PCA component 1")
ax.set_ylabel("PCA component 2")
ax.set_zlabel("PCA component 3")
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# Display the plot
plt.show()

