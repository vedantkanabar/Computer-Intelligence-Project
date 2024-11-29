import pandas as pd

EU_Countries_Code = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE",
    "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL",
    "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL",
    "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
]

EU_Countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

# Create a mapping from country code to Country name
country_code_mapping = dict(zip(EU_Countries_Code, EU_Countries))

try:
    df = pd.read_excel('human-freedom-index-2023-datafile.xlsx')
except FileNotFoundError:
    print("The file cannot be opened")
    exit()

# Exract data for EU countries from 2012-2021
filtered_EU_countries = df[df.iloc[:, 1].isin(EU_Countries_Code) & df.iloc[:, 0].between(2012, 2021)]

# Replace country names to match what has been saved in other datasets
filtered_EU_countries.iloc[:, 2] = (filtered_EU_countries.iloc[:, 1]).map(country_code_mapping)

# Set up missing column names
filtered_EU_countries.columns.values[0] = "year"
filtered_EU_countries.columns.values[1] = "country_code"
filtered_EU_countries.columns.values[2] = "country_name"
filtered_EU_countries.columns.values[3] = "country_region"

# Convert columns to snake case
filtered_EU_countries.columns = [col.lower().replace(" ", "_") for col in filtered_EU_countries.columns]

# Drop column country_region which is not needed
filtered_EU_countries = filtered_EU_countries.drop(columns=["country_region"])


# Function convert_range_to_average to convert a range to an average
# Inputs:
#   - value, a string containing a range e.g. (50-55)
# Ouputs:
#   - a float which is the average or midpoint of the range.
def convert_range_to_average(value):
    # Check if the value contains a range (e.g., '50-55')
    if isinstance(value, str) and '-' in value:
        # Split the range and calculate the average
        lower, upper = value.split('-')
        return (float(lower) + float(upper)) / 2
    # Return the value as is if it's not a range
    return value

# Convert all ranges to a midpoint value
filtered_EU_countries = filtered_EU_countries.map(convert_range_to_average)

# Save the cleaned and reshaped data to a new CSV file
filtered_EU_countries.to_csv("cleaned_human_freedom_idx.csv", index=False)