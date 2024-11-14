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
    df = pd.read_excel('human-freedom-index-2023-datafile.xlsx')
except FileNotFoundError:
    print("The file cannot be opened")
    exit()

filtered_EU_countries = df[df.iloc[:, 1].isin(EU_Countries_Code) & df.iloc[:, 0].between(2012, 2021)]
filtered_EU_countries.iloc[:, 2] = (filtered_EU_countries.iloc[:, 1]).map(country_code_mapping)
filtered_EU_countries.columns.values[0] = "year"
filtered_EU_countries.columns.values[1] = "country_code"
filtered_EU_countries.columns.values[2] = "country_name"
filtered_EU_countries.columns.values[3] = "country_region"
filtered_EU_countries.columns = [col.lower().replace(" ", "_") for col in filtered_EU_countries.columns]

filtered_EU_countries.to_csv("cleaned_human_freedom_idx.csv", index=False)