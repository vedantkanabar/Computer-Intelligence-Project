from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from utils import extract_features

feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

# Calculate distance matrix for MDS
distance_matrix = squareform(pdist(feature_array, metric="euclidean"))

# Apply MDS for dimensionality reduction to 3 components
mds = MDS(n_components=3, dissimilarity="precomputed")
mds_result = mds.fit_transform(distance_matrix)

# Plot the MDS results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(mds_result[:, 0], mds_result[:, 1], mds_result[:, 2], c=happiness_index_array, cmap="viridis", edgecolor="k", s=50)

ax.set_title("3D Visualization of MDS Reduced Features (One year back model)")
ax.set_xlabel("MDS component 1")
ax.set_ylabel("MDS component 2")
ax.set_zlabel("MDS component 3")
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# Display the plot
plt.show()