import pandas as pd
import numpy as np
import json

# Read the CSV file
df = pd.read_csv('combined_dataset.csv')

# Initialize a list to store JSON data
json_data = []

# Iterate over each row to build the JSON structure
for _, row in df.iterrows():

    # Extract current row
    country_name = row["country_name"]
    year = int(row["year"])

    year_back = df.loc[(df["country_name"] == country_name) & (df["year"] == year - 1)]

    # Incase of years with no data back from 1 year back (e.g. 2012) continue to next row
    if year_back.empty:
        continue

    # # Create the base JSON structure with country and year
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

    year_back.columns = [col+"_1_back" for col in year_back.columns]
    year_back_row = year_back.iloc[0]

    # Iterate over each column in the row 1 year back, except for the ones we already used in the labels
    for column in year_back.columns:
        if column not in ["country_name_1_back", "year_1_back", "happiness_index_1_back"]:
            # Add the column value to the "features" dictionary
            if isinstance(year_back_row[column], np.integer):
                data["features"][column] = int(year_back_row[column])
            else:
                data["features"][column] = year_back_row[column]

    
    # Append the data for the current row to the JSON list
    json_data.append(data)

# Convert the list of dictionaries to a JSON object
json_output = json.dumps(json_data, indent=4)

with open('feature_year_back.json', 'w') as f:
    f.write(json_output)