import pandas as pd

# Load the CSV file
df = pd.read_csv("EUCountriesDevelopmentIndicators.csv")

# Replace country names
df['country_name'] = df['country_name'].replace({
    'Cyprus': 'Republic of Cyprus',
    'Slovak Republic': 'Slovakia',
    'Czechia': 'Czech Republic'
})

# Melt the DataFrame to have years in a single column
df_melted = df.melt(id_vars=["country_name", "Country Code", "Series Name"], 
                    value_vars=["2012 [YR2012]", "2013 [YR2013]", "2014 [YR2014]", 
                                "2015 [YR2015]", "2016 [YR2016]", "2017 [YR2017]",
                                "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]",
                                "2021 [YR2021]", "2022 [YR2022]"],
                    var_name="year", 
                    value_name="value")

# Clean up the `year` column to remove `[YRxxxx]` and keep only the year part
df_melted['year'] = df_melted['year'].str.replace(r'\s*\[YR\d{4}\]', '', regex=True)

# Pivot the table so each "Series Name" becomes a column, with year as the columns
df_pivot = df_melted.pivot_table(index=["country_name", "year"], 
                                 columns="Series Name", 
                                 values="value", 
                                 aggfunc="first").reset_index()

df_pivot.columns.name = None

# Convert column names to snake case
df_pivot.columns = [col.lower().replace(" ", "_") for col in df_pivot.columns]

# Save the cleaned and reshaped data to a new CSV file
df_pivot.to_csv("cleaned_EUCountriesDevelopmentIndicators.csv", index=False)
