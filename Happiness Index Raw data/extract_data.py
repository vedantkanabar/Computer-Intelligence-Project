import pandas as pd

EU_Countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]


try:
    data_df = pd.read_excel('DataForTable2.1.xls')
except FileNotFoundError:
    print("The file data.csv was not found in the specified location.")
    exit()

filtered_countries_df = data_df[data_df['Country name'].isin(EU_Countries) 
                                & (data_df['year'].between(2012, 2022))][['Country name', 'year', 'Life Ladder']]
print(filtered_countries_df.head())

filtered_countries_df = filtered_countries_df.rename(columns={'Country name': 'country_name'})
filtered_countries_df = filtered_countries_df.rename(columns={'Life Ladder': 'happiness_index'})

filtered_countries_df.to_csv("Cleaned_Happiness_index.csv", index=False)