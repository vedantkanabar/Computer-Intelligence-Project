import pandas as pd

df = pd.read_csv("EUCountriesDevelopmentIndicators.csv")

# Replace country names
df['country_name'] = df['country_name'].replace({
    'Cyprus': 'Republic of Cyprus',
    'Slovak Republic': 'Slovakia',
    'Czechia': 'Czech Republic'
})

# Save the cleaned data to a new CSV file
df.to_csv("cleaned_EUCountriesDevelopmentIndicators.csv", index=False)