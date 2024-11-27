from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from utils import extract_features_label

EU_Countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

# Extract features from json file
feature_array, feature_label, feature_columns, happiness_index_array = extract_features_label('../Combining Data and Feature creation/feature.json')

mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0)

# fit our ditance matrix into the mds function
mds_result = mds.fit_transform(feature_array)

# Create labels refering to datapoints
colors = [color_map[label] for label in labels]

# Plot the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot each label by itself
for label in color_map.keys():

    # Select valid indexes for label with np.where(
    indices = np.where(labels == label)

    # Add those data points with the correct label
    ax.scatter(mds_result[indices, 0], mds_result[indices, 1], mds_result[indices, 2], label=label, color=color_map[label], s=10)

# Setting labels and title
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title(f"3D MDS Representation for {k}-mers")

# Setting legend
ax.legend()

# UNCOMMENT line if you want to play around with 3d image
plt.show()