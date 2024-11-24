import pandas as pd

EU_Countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Republic of Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

print(len(EU_Countries))


try:
    data_df = pd.read_excel('DataForTable2.1.xls')
except FileNotFoundError:
    print("The file data.csv was not found in the specified location.")
    exit()

data_df['Country name'] = data_df['Country name'].replace({
    'Cyprus': 'Republic of Cyprus',
    'Czechia': 'Czech Republic'
})

filtered_countries_df = data_df[data_df['Country name'].isin(EU_Countries) 
                                & (data_df['year'].between(2012, 2021))]

filtered_countries_df = filtered_countries_df.rename(columns={'Life Ladder': 'happiness_index'})

# Convert column names to snake case
filtered_countries_df.columns = [col.lower().replace(" ", "_") for col in filtered_countries_df.columns]

filtered_countries_df.to_csv("cleaned_happiness_index.csv", index=False)

print(filtered_countries_df.shape)