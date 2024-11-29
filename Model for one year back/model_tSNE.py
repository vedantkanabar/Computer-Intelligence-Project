from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import extract_features

feature_array, feature_columns, happiness_index_array = extract_features("../Combining Data and Feature creation/feature_year_back.json")

# Apply tSNE for dimensionality reduction to 3 components
tsne = TSNE(n_components=3)
tsne_result = tsne.fit_transform(feature_array)

# Plot the tSNE results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=happiness_index_array, cmap="viridis", edgecolor="k", s=50)

ax.set_title("3D Visualization of t-SNE Reduced Features (One year back model)")
ax.set_xlabel("t-SNE component 1")
ax.set_ylabel("t-SNE component 2")
ax.set_zlabel("t-SNE component 3")
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# Display the plot
plt.show()

