import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('combined_dataset.csv')

# Initialize a list to store JSON data
json_data = []

# Iterate over each row to build the JSON structure
for _, row in df.iterrows():
    # Create the base JSON structure with country and year
    data = {
        "labels": {
            "country": row["country_name"],
            "year": row["year"]
        },
        "happiness_index": row["happiness_index"],
        "features": {}
    }
    
    # Iterate over each column in the row, except for the ones we already used in the labels
    for column in df.columns:
        if column not in ["country_name", "year", "happiness_index"]:
            # Add the column value to the "features" dictionary
            data["features"][column] = row[column]
    
    # Append the data for the current row to the JSON list
    json_data.append(data)

# Convert the list of dictionaries to a JSON object
json_output = json.dumps(json_data, indent=4)

with open('feature.json', 'w') as f:
    f.write(json_output)