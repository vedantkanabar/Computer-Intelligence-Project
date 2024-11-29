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
    df = pd.read_csv('economicdata2012-2022.csv', header=4)
except FileNotFoundError:
    print("The file cannot be opened")
    exit()

# Exract data for EU countries from 2012-2021
filtered_EU_countries = df[df['ISO_Code'].isin(EU_Countries_Code) & df['Year'].between(2012, 2021)]

# Drop the first column which is empty
filtered_EU_countries = filtered_EU_countries.drop(df.columns[0], axis=1)

# Create a new colum Countires with the Country name from mapping of country code
filtered_EU_countries.loc[:, 'Countries'] = (filtered_EU_countries['ISO_Code']).map(country_code_mapping)

# Convert the column names to match other table and to be snake case
filtered_EU_countries = filtered_EU_countries.rename(columns={'Countries': 'country_name', 'ISO_Code': 'country_code'})
filtered_EU_countries.columns = [col.lower().replace(" ", "_").replace(".", "_") for col in filtered_EU_countries.columns]


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
filtered_EU_countries.to_csv("cleaned_economic_data.csv", index=False)