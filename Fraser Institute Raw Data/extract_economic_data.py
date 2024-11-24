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

country_code_mapping = dict(zip(EU_Countries_Code, EU_Countries))

try:
    df = pd.read_csv('economicdata2012-2022.csv', header=4)
except FileNotFoundError:
    print("The file cannot be opened")
    exit()

filtered_EU_countries = df[df['ISO_Code'].isin(EU_Countries_Code) & df['Year'].between(2012, 2021)]
filtered_EU_countries = filtered_EU_countries.drop(df.columns[0], axis=1)
filtered_EU_countries.loc[:, 'Countries'] = (filtered_EU_countries['ISO_Code']).map(country_code_mapping)
filtered_EU_countries = filtered_EU_countries.rename(columns={'Countries': 'country_name', 'ISO_Code': 'country_code'})
filtered_EU_countries.columns = [col.lower().replace(" ", "_").replace(".", "_") for col in filtered_EU_countries.columns]

def convert_range_to_average(value):
    # Check if the value contains a range (e.g., '50-55')
    if isinstance(value, str) and '-' in value:
        # Split the range and calculate the average
        lower, upper = value.split('-')
        return (float(lower) + float(upper)) / 2
    # Return the value as is if it's not a range
    return value

filtered_EU_countries = filtered_EU_countries.map(convert_range_to_average)

filtered_EU_countries.to_csv("cleaned_economic_data.csv", index=False)